import subprocess
from pathlib import Path
from contextlib import ExitStack
import argparse
import json
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from matplotlib.tri import Triangulation
import numpy as np

case = Path(__file__).resolve().parent

def read_foam_list(path):
    text = path.read_text()
    match = re.search(r'\n\s*(\d+)\s*\n\(\s*\n', text)
    if match is None:
        raise ValueError('Expected an ASCII OpenFOAM list: ' + str(path))
    return text[match.end():text.rfind(')')]

def read_patch(text, name):
    match = re.search(r'\b' + re.escape(name) + r'\s*\{([^{}]*)\}', text)
    if match is None:
        raise ValueError('Missing boundary patch: ' + name)
    return match.group(1)

def read_values(text, keyword, count, components):
    match = re.search(r'\b' + keyword + r'\s+uniform\s+([^;]+);', text)
    if match is not None:
        value = np.fromstring(match.group(1).replace('(', '').replace(')', ''), sep=' ')
        if value.size != components:
            raise ValueError('Unexpected uniform field size')
        return np.tile(value, (count, 1)).reshape(count, components)
    pattern = r'\b' + keyword + r'\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;'
    match = re.search(pattern, text, re.S)
    if match is None:
        raise ValueError('Expected an ASCII scalar or vector field: ' + keyword)
    values = np.fromstring(match.group(2).replace('(', ' ').replace(')', ' '), sep=' ')
    if int(match.group(1)) != count or values.size != count * components:
        raise ValueError('Field size does not match the mesh')
    if not np.all(np.isfinite(values)):
        raise ValueError('Non-finite field values')
    return values.reshape(count, components)

def read_mesh():
    folder = case / 'constant' / 'polyMesh'
    text = read_foam_list(folder / 'points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ').reshape(-1, 3)
    owners = np.fromstring(read_foam_list(folder / 'owner'), sep=' ', dtype=int)
    faces = read_foam_list(folder / 'faces').splitlines()
    boundary = (folder / 'boundary').read_text()
    patch = read_patch(boundary, 'frontAndBack')
    first = int(re.search(r'startFace\s+(\d+)', patch).group(1))
    count = int(re.search(r'nFaces\s+(\d+)', patch).group(1))
    quads = re.findall(r'4\s*\(([^)]+)\)', '\n'.join(faces[first:first + count]))
    if len(quads) != count:
        raise ValueError('Expected quadrilateral front and back faces')
    quads = np.fromstring(' '.join(quads), sep=' ', dtype=int).reshape(-1, 4)
    bottom = np.all(np.isclose(points[quads, 2], points[:, 2].min()), axis=1)
    cell_ids = owners[first:first + count][bottom]
    quads = quads[bottom]
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    triangle_points = points[triangles, :2]
    first_edge = triangle_points[:, 1] - triangle_points[:, 0]
    second_edge = triangle_points[:, 2] - triangle_points[:, 0]
    cross_product = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    clockwise = cross_product < 0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    used, inverse = np.unique(triangles, return_inverse=True)
    triangulation = Triangulation(points[used, 0], points[used, 1], inverse.reshape(-1, 3))
    triangle_cells = np.concatenate([cell_ids, cell_ids])
    nodes = points[used, :2]
    local_quads = np.searchsorted(used, quads)
    centres = points[quads, :2].mean(axis=1)
    distances = np.linalg.norm(points[quads, :2] - centres[:, None, :], axis=2)
    weights = 1 / np.maximum(distances, 1e-12)
    total_weights = np.bincount(local_quads.ravel(), weights=weights.ravel(), minlength=len(nodes))
    coordinates = nodes[triangulation.triangles]
    x1, x2, x3 = coordinates[:, :, 0].T
    y1, y2, y3 = coordinates[:, :, 1].T
    determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    derivative_x = np.column_stack([y2 - y3, y3 - y1, y1 - y2]) / determinant[:, None]
    derivative_y = np.column_stack([x3 - x2, x1 - x3, x2 - x1]) / determinant[:, None]
    shape = json.loads((case / 'mesh_design.json').read_text()).get('shape', 'cylinder')
    patch = read_patch(boundary, shape)
    first = int(re.search(r'startFace\s+(\d+)', patch).group(1))
    count = int(re.search(r'nFaces\s+(\d+)', patch).group(1))
    wall_centres = []
    for line in faces[first:first + count]:
        indices = np.fromstring(line[line.index('(') + 1:line.index(')')], sep=' ', dtype=int)
        wall_centres.append(points[indices, :2].mean(axis=0))
    wall_centres = np.asarray(wall_centres)
    theta = np.mod(np.arctan2(wall_centres[:, 1], wall_centres[:, 0]), 2 * np.pi)
    return {
        'cell_count': int(owners.max()) + 1,
        'wall_count': count,
        'wall_owners': owners[first:first + count],
        'theta': theta,
        'find_triangle': triangulation.get_trifinder(),
        'triangle_cells': triangle_cells,
        'nodes': nodes,
        'quads': local_quads,
        'cell_ids': cell_ids,
        'weights': weights,
        'total_weights': total_weights,
        'triangles': triangulation.triangles,
        'derivative_x': derivative_x,
        'derivative_y': derivative_y,
    }

def read_force_history():
    rows = {}
    paths = sorted((case / 'postProcessing' / 'forceCoefficients').glob('*/forceCoeffs.dat'),
                   key=lambda path: float(path.parent.name))
    if not paths:
        raise ValueError('No forceCoeffs.dat found')
    for path in paths:
        data = np.loadtxt(path, comments='#', ndmin=2)
        for row in data:
            rows[row[0]] = row[:4]
    return np.asarray([rows[time] for time in sorted(rows)])

def read_simulation():
    times = []
    for folder in case.iterdir():
        if folder.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?', folder.name):
            if (folder / 'U').is_file() and (folder / 'p').is_file():
                times.append((float(folder.name), folder))
    times.sort()
    if len(times) < 2:
        raise ValueError('Wait for field reconstruction, or run reconstructPar -newTimes after the solver finishes.')
    control = (case / 'system' / 'controlDict').read_text()
    speed = float(re.search(r'magUInf\s+([\d.eE+-]+)', control).group(1))
    design = json.loads((case / 'mesh_design.json').read_text())
    shape = design.get('shape', 'cylinder')
    length_name = 'diameter'
    if shape == 'square':
        length_name = 'frontal_width'
    diameter = float(design[length_name])
    properties = (case / 'constant' / 'physicalProperties').read_text()
    viscosity = float(re.search(r'\bnu\s+([\d.eE+-]+)', properties).group(1))
    outlet = read_patch((case / '0' / 'p').read_text(), 'outlet')
    reference_pressure = float(re.search(r'value\s+uniform\s+([\d.eE+-]+)', outlet).group(1))
    return {
        'shape': shape,
        'times': times,
        'forces': read_force_history(),
        'mesh': read_mesh(),
        'speed': speed,
        'diameter': diameter,
        'reynolds': speed * diameter / viscosity,
        'reference_pressure': reference_pressure,
    }

def time_average(times, values):
    intervals = np.diff(times)
    weights = intervals.reshape((-1,) + (1,) * (values.ndim - 1))
    integral = np.sum(0.5 * (values[:-1] + values[1:]) * weights, axis=0)
    return integral / (times[-1] - times[0])

def select_average_times(simulation, start, end):
    selected = [(time, folder) for time, folder in simulation['times'] if start <= time <= end]
    if len(selected) < 2:
        raise ValueError('The averaging interval needs at least two reconstructed fields')
    return selected

def analyze_surface_pressure(simulation, average_times, output):
    mesh = simulation['mesh']
    pressure_history = []
    for index, (time, folder) in enumerate(average_times):
        text = (folder / 'p').read_text()
        wall = read_patch(text, simulation['shape'])
        if re.search(r'\bvalue\s+', wall):
            pressure = read_values(wall, 'value', mesh['wall_count'], 1)[:, 0]
        elif re.search(r'\btype\s+zeroGradient\s*;', wall):
            cell_pressure = read_values(text, 'internalField', mesh['cell_count'], 1)[:, 0]
            pressure = cell_pressure[mesh['wall_owners']]
        else:
            raise ValueError('Unsupported cylinder pressure boundary condition')
        cp = 2 * (pressure - simulation['reference_pressure']) / simulation['speed'] ** 2
        pressure_history.append(cp)
        if index % 50 == 0:
            print(f'Reading surface pressure: {index + 1}/{len(average_times)}', flush=True)
    times = np.asarray([time for time, folder in average_times])
    history = np.asarray(pressure_history)
    mean = time_average(times, history)
    variance = time_average(times, (history - mean) ** 2)
    deviation = np.sqrt(variance)
    order = np.argsort(mesh['theta'])
    angles = np.rad2deg(mesh['theta'][order])
    table = np.column_stack([angles, mean[order], deviation[order]])
    np.savetxt(output / 'surface_cp.csv', table, delimiter=',',
               header='theta_degrees,cp_mean,cp_standard_deviation', comments='')
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.plot(angles, mean[order], color='#176d9b', label='OpenFOAM: time mean')
    ax.fill_between(angles, (mean - deviation)[order], (mean + deviation)[order],
                    color='#176d9b', alpha=0.18, label='OpenFOAM: mean +/- standard deviation')
    ax.set(xlabel='Angle from +x, counterclockwise [deg]', ylabel='$C_p$', xlim=(0, 360),
           title=f'{simulation["shape"].capitalize()} | Re = {simulation["reynolds"]:g} | t = {times[0]:g}-{times[-1]:g} s')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / 'surface_cp.png', dpi=180)
    plt.close(fig)

def analyze_forces(simulation, average_times, output):
    history = simulation['forces']
    start = average_times[0][0]
    end = average_times[-1][0]
    if history[0, 0] > start or history[-1, 0] < end:
        raise ValueError('The force history does not cover the averaging interval')
    interior = history[(history[:, 0] > start) & (history[:, 0] < end)]
    first = np.array([start] + [np.interp(start, history[:, 0], history[:, i]) for i in range(1, 4)])
    last = np.array([end] + [np.interp(end, history[:, 0], history[:, i]) for i in range(1, 4)])
    selected_forces = np.vstack([first, interior, last])
    times = selected_forces[:, 0]
    coefficients = selected_forces[:, 1:4]
    mean = time_average(times, coefficients)
    deviation = np.sqrt(time_average(times, (coefficients - mean) ** 2))
    summary = {
        'reynolds': simulation['reynolds'],
        'average_start_seconds': start,
        'average_end_seconds': end,
        'pressure_snapshots': len(average_times),
        'cm_mean': float(mean[0]),
        'cd_mean': float(mean[1]),
        'cl_mean': float(mean[2]),
        'cm_standard_deviation': float(deviation[0]),
        'cd_standard_deviation': float(deviation[1]),
        'cl_standard_deviation': float(deviation[2]),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savetxt(output / 'force_history.csv', history, delimiter=',',
               header='time_seconds,cm,cd,cl', comments='')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    for ax, column, label in [(axes[0], 2, '$C_D$'), (axes[1], 3, '$C_L$')]:
        ax.plot(history[:, 0], history[:, column], color='#176d9b', label='OpenFOAM')
        ax.axvspan(start, end, color='#176d9b', alpha=0.12, label='Statistics interval')
        ax.set(ylabel=label, xlim=(history[0, 0], history[-1, 0]))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{simulation["shape"].capitalize()} | Re = {simulation["reynolds"]:g}')
    fig.savefig(output / 'force_history.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)

def prepare_video_grid(simulation):
    diameter = simulation['diameter']
    bounds = np.array([-2.0, 10.0, -3.0, 3.0]) * diameter
    x0, x1, y0, y1 = bounds
    x = np.linspace(x0, x1, 900)
    y = np.linspace(y0, y1, 450)
    xx, yy = np.meshgrid(x, y)
    triangles = simulation['mesh']['find_triangle'](xx.ravel(), yy.ravel())
    solid = inside_body(np.column_stack([xx.ravel(), yy.ravel()]), 0.5 * diameter, simulation['shape']).reshape(xx.shape)
    if np.any((triangles.reshape(xx.shape) < 0) & ~solid):
        raise ValueError('The video window extends outside the fluid mesh')
    cell_ids = simulation['mesh']['triangle_cells'][np.maximum(triangles, 0)]
    return {
        'x': xx,
        'y': yy,
        'bounds': bounds,
        'solid': solid,
        'cell_ids': cell_ids,
        'triangles': np.maximum(triangles, 0),
    }

def calculate_vorticity(node_velocity, mesh, triangles):
    vertices = mesh['triangles'][triangles]
    dv_dx = np.sum(node_velocity[vertices, 1] * mesh['derivative_x'][triangles], axis=1)
    du_dy = np.sum(node_velocity[vertices, 0] * mesh['derivative_y'][triangles], axis=1)
    return dv_dx - du_dy

def read_video_fields(folder, simulation, grid):
    mesh = simulation['mesh']
    values = read_values((folder / 'U').read_text(), 'internalField', mesh['cell_count'], 3)[:, :2]
    velocity = values[grid['cell_ids']].reshape(grid['x'].shape + (2,))
    velocity[grid['solid']] = 0
    weighted = values[mesh['cell_ids'], None, :] * mesh['weights'][:, :, None]
    node_velocity = np.empty((len(mesh['nodes']), 2))
    for component in range(2):
        total = np.bincount(mesh['quads'].ravel(), weights=weighted[:, :, component].ravel(),
                            minlength=len(mesh['nodes']))
        node_velocity[:, component] = total / mesh['total_weights']
    radius = 0.5 * simulation['diameter']
    distance = np.linalg.norm(mesh['nodes'], axis=1)
    if simulation['shape'] == 'square':
        distance = np.abs(mesh['nodes']).sum(axis=1)
    wall = np.isclose(distance, radius, atol=1e-10, rtol=0)
    node_velocity[wall] = 0
    omega = calculate_vorticity(node_velocity, mesh, grid['triangles']).reshape(grid['x'].shape)
    omega *= simulation['diameter'] / simulation['speed']
    return velocity, omega

def interpolate_particle_velocity(positions, velocity, bounds):
    x0, x1, y0, y1 = bounds
    height, width = velocity.shape[:2]
    x = (positions[:, 0] - x0) * (width - 1) / (x1 - x0)
    y = (positions[:, 1] - y0) * (height - 1) / (y1 - y0)
    x = np.clip(x, 0, width - 1.000001)
    y = np.clip(y, 0, height - 1.000001)
    ix = x.astype(int)
    iy = y.astype(int)
    fraction_x = (x - ix)[:, None]
    fraction_y = (y - iy)[:, None]
    lower = (1 - fraction_x) * velocity[iy, ix] + fraction_x * velocity[iy, ix + 1]
    upper = (1 - fraction_x) * velocity[iy + 1, ix] + fraction_x * velocity[iy + 1, ix + 1]
    return (1 - fraction_y) * lower + fraction_y * upper

def inside_body(positions, radius, shape):
    if shape == 'square':
        return np.abs(positions).sum(axis=1) <= radius
    return np.sum(positions * positions, axis=1) <= radius * radius

def seed_particles(bounds, radius, random, shape):
    x0, x1, y0, y1 = bounds
    positions = np.column_stack([random.uniform(x0, x1, 1200), random.uniform(y0, y1, 1200)])
    inside = inside_body(positions, radius, shape)
    positions[inside, 0] = x0
    return positions

def advance_particles(positions, velocity, dt, grid, radius, random, shape):
    bounds = grid['bounds']
    first = interpolate_particle_velocity(positions, velocity, bounds)
    midpoint = positions + 0.5 * dt * first
    second = interpolate_particle_velocity(midpoint, velocity, bounds)
    positions += dt * second
    x0, x1, y0, y1 = bounds
    reset = (positions[:, 0] < x0) | (positions[:, 0] > x1)
    reset |= (positions[:, 1] < y0) | (positions[:, 1] > y1)
    reset |= inside_body(positions, radius, shape)
    positions[reset, 0] = x0
    positions[reset, 1] = random.uniform(y0, y1, int(reset.sum()))

def create_video_axes(simulation, grid, name, color_limit, fig=None, ax=None):
    cmap = 'magma'
    minimum = 0
    maximum = color_limit
    label = r'$|\mathbf{u}|/U_\infty$'
    if name == 'vorticity':
        cmap = LinearSegmentedColormap.from_list('vorticity',
            ['#67d7f0', '#1e6fb2', '#101724', '#c72b4d', '#ffd185'])
        minimum = -color_limit
        label = r'$\omega_z D/U_\infty$'
    if fig is None:
        fig, ax = plt.subplots(figsize=(14, 7.5), layout='constrained', facecolor='#101724')
    ax.set_facecolor('#101724')
    bounds = grid['bounds'] / simulation['diameter']
    image = ax.imshow(np.ma.array(np.zeros_like(grid['x']), mask=grid['solid']),
                      origin='lower', extent=bounds, cmap=cmap, vmin=minimum, vmax=maximum,
                      interpolation='nearest', aspect='equal')
    if simulation['shape'] == 'square':
        ax.add_patch(Polygon([(-0.5, 0), (0, 0.5), (0.5, 0), (0, -0.5)], color='#dce3ed', zorder=4))
    else:
        ax.add_patch(Circle((0, 0), 0.5, color='#dce3ed', zorder=4))
    scatter = ax.scatter([], [], s=4, c='white', alpha=0.65, linewidths=0, zorder=3)
    coefficients = ax.text(0.98, 0.95, '', transform=ax.transAxes, ha='right', va='top',
                           color='white', bbox={'facecolor': '#101724', 'alpha': 0.85, 'edgecolor': 'none'})
    length_symbol = 'D'
    if simulation['shape'] == 'square':
        length_symbol = 'H'
        label = label.replace(' D/', ' H/')
    ax.set(xlabel='x/' + length_symbol, ylabel='y/' + length_symbol, xlim=bounds[:2], ylim=bounds[2:])
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='#dce3ed')
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.015, extend='both')
    colorbar.set_label(label, color='white')
    colorbar.ax.tick_params(colors='#dce3ed')
    clock = ax.set_title(f'{simulation["shape"].capitalize()} | {name} | Re = {simulation["reynolds"]:g} | t = 0.00 s',
                         color='white', fontsize=14)
    return fig, image, scatter, coefficients, clock

def create_videos(simulation, arguments, output):
    if shutil.which('ffmpeg') is None:
        raise ValueError('ffmpeg is required to create the videos')
    grid = prepare_video_grid(simulation)
    times = np.asarray([time for time, folder in simulation['times']])
    start = max(arguments.video_start, times[0])
    end = times[-1]
    if arguments.video_end is not None:
        end = min(arguments.video_end, end)
    if end <= start:
        raise ValueError('The video interval must contain at least two distinct times')
    duration = end - start
    if arguments.video_duration is not None:
        duration = arguments.video_duration
    frame_count = max(2, int(round(duration * arguments.fps)) + 1)
    frame_times = np.linspace(start, end, frame_count)
    radius = 0.5 * simulation['diameter']
    random = np.random.default_rng(7)
    positions = seed_particles(grid['bounds'], radius, random, simulation['shape'])
    names = ['velocity', 'vorticity']
    limits = [arguments.velocity_limit, arguments.vorticity_limit]
    views = []
    previous_time = start
    cached_index = -1
    current = None
    following = None
    forces = simulation['forces']
    with ExitStack() as stack:
        combined_writer = None
        combined_fig = None
        axes = [None, None]
        if arguments.readme_gif or arguments.stacked_video:
            combined_fig, axes = plt.subplots(2, 1, figsize=(10, 10.4), layout='constrained',
                                             facecolor='#101724')
            stack.callback(plt.close, combined_fig)
        for name, limit, ax in zip(names, limits, axes):
            fig, image, scatter, label, clock = create_video_axes(simulation, grid, name, limit, combined_fig, ax)
            stack.callback(plt.close, fig)
            writer = FFMpegWriter(fps=arguments.fps, codec='libx264', extra_args=[
                '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                '-movflags', '+faststart'])
            if arguments.readme_gif or arguments.stacked_video:
                if combined_writer is None:
                    combined_writer = writer
                    stack.enter_context(writer.saving(fig, str(output / (simulation['shape'] + '.mp4')), dpi=90))
            else:
                stack.enter_context(writer.saving(fig, str(output / (name + '.mp4')), dpi=100))
            views.append((writer, fig, image, scatter, label, clock))
        for figure in {view[1] for view in views}:
            figure.canvas.draw()
            figure.set_layout_engine(None)
        for frame, time in enumerate(frame_times):
            index = int(np.searchsorted(times, time, side='right')) - 1
            index = max(0, min(index, len(times) - 2))
            if index != cached_index:
                if index == cached_index + 1 and following is not None:
                    current = following
                else:
                    current = read_video_fields(simulation['times'][index][1], simulation, grid)
                following = read_video_fields(simulation['times'][index + 1][1], simulation, grid)
                cached_index = index
            fraction = (time - times[index]) / (times[index + 1] - times[index])
            velocity = (1 - fraction) * current[0] + fraction * following[0]
            omega = (1 - fraction) * current[1] + fraction * following[1]
            dt = time - previous_time
            substeps = max(1, int(np.ceil(dt * simulation['speed'] / (0.02 * simulation['diameter']))))
            for substep in range(substeps):
                advance_particles(positions, velocity, dt / substeps, grid, radius, random, simulation['shape'])
            fields = [np.linalg.norm(velocity, axis=2) / simulation['speed'], omega]
            cd = np.interp(time, forces[:, 0], forces[:, 2])
            cl = np.interp(time, forces[:, 0], forces[:, 3])
            for name, values, view in zip(names, fields, views):
                writer, fig, image, scatter, label, clock = view
                image.set_data(np.ma.array(values, mask=grid['solid']))
                scatter.set_offsets(positions / simulation['diameter'])
                label.set_text(f'Cd = {cd:.3f}\nCl = {cl:+.3f}')
                clock.set_text(f'{simulation["shape"].capitalize()} | {name} | Re = {simulation["reynolds"]:g} | t = {time:.2f} s')
                if not (arguments.readme_gif or arguments.stacked_video):
                    writer.grab_frame()
            if combined_writer is not None:
                combined_writer.grab_frame()
            previous_time = time
            if frame % 100 == 0:
                print(f'Videos: {frame + 1}/{frame_count} frames', flush=True)

    if arguments.readme_gif:
        destination = case.parents[1] / 'media'
        destination.mkdir(exist_ok=True)
        palette = output / (simulation['shape'] + '_palette.png')
        subprocess.run(['ffmpeg', '-y', '-threads', '1', '-i', str(output / (simulation['shape'] + '.mp4')),
                        '-vf', 'fps=20,scale=640:-1:flags=lanczos,palettegen=max_colors=64',
                        '-frames:v', '1', '-threads', '1', str(palette)], check=True)
        subprocess.run(['ffmpeg', '-y', '-threads', '1', '-i', str(output / (simulation['shape'] + '.mp4')),
                        '-i', str(palette), '-filter_complex_threads', '1', '-filter_complex',
                        '[0:v]fps=20,scale=640:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3',
                        '-loop', '0', str(destination / (simulation['shape'] + '.gif'))], check=True)
        palette.unlink()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Postprocess the Navier-Stokes cylinder simulation.')
    parser.add_argument('--case', type=Path, default=case, help='Case directory; defaults to the cylinder.')
    parser.add_argument('--stacked-video', action='store_true', help='Create one video with speed above vorticity.')
    parser.add_argument('--videos-only', action='store_true', help='Skip force and pressure plots.')
    parser.add_argument('--average-start', type=float, default=30.0, help='Start of the time average, in physical seconds.')
    parser.add_argument('--average-end', type=float, help='End of the time average; defaults to the latest saved time.')
    parser.add_argument('--readme-gif', action='store_true', help='Create a stacked velocity/vorticity GIF in media/.')
    parser.add_argument('--plots-only', action='store_true', help='Save figures and numerical results without encoding the videos.')
    parser.add_argument('--video-start', type=float, default=0.0)
    parser.add_argument('--video-end', type=float)
    parser.add_argument('--video-duration', type=float, help='Video duration in seconds; defaults to physical duration.')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--velocity-limit', type=float, default=2.0)
    parser.add_argument('--vorticity-limit', type=float, default=4.0)
    arguments = parser.parse_args()
    if arguments.fps <= 0:
        parser.error('--fps must be positive')
    if arguments.video_duration is not None and arguments.video_duration <= 0:
        parser.error('--video-duration must be positive')
    if min(arguments.velocity_limit, arguments.vorticity_limit) <= 0:
        parser.error('Color limits must be positive')
    return arguments

def main():
    global case
    arguments = parse_arguments()
    case = arguments.case.resolve()
    simulation = read_simulation()

    average_end = simulation['times'][-1][0]
    if arguments.average_end is not None:
        average_end = arguments.average_end
    average_times = select_average_times(simulation, arguments.average_start, average_end)
    output = case / 'results'
    output.mkdir(exist_ok=True)

    if not arguments.videos_only:
        analyze_surface_pressure(simulation, average_times, output)
        analyze_forces(simulation, average_times, output)
    if not arguments.plots_only:
        create_videos(simulation, arguments, output)
    print('Postprocessing saved in ' + str(output), flush=True)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
