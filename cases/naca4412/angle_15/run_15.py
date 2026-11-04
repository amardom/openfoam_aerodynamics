from pathlib import Path
import argparse
import fcntl
import json
import math
import re
import shlex
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

case = Path(__file__).resolve().parent

def mesh_vertex(x, y, z, vertices, lookup):
    key = (round(x, 13), round(y, 13), z)
    if key not in lookup:
        lookup[key] = len(vertices)
        vertices.append(key)
    return lookup[key]

def profile_point(x, upper, rotation):
    m = 0.04
    p = 0.4
    thickness = 0.12
    half_thickness = 5 * thickness * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    if x < p:
        camber = m / (p * p) * (2 * p * x - x * x)
        slope = 2 * m / (p * p) * (p - x)
    else:
        camber = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x * x)
        slope = 2 * m / (1 - p) ** 2 * (p - x)
    theta = math.atan(slope)
    if upper:
        sign = 1
    else:
        sign = -1
    surface_x = x - sign * half_thickness * math.sin(theta)
    surface_y = camber + sign * half_thickness * math.cos(theta)
    offset_x = surface_x - 0.25
    rotated_x = 0.25 + offset_x * math.cos(rotation) - surface_y * math.sin(rotation)
    rotated_y = offset_x * math.sin(rotation) + surface_y * math.cos(rotation)
    return (rotated_x, rotated_y)

def add_mesh_block(points, nx, ny, gx, gy, boundary, vertices, lookup, blocks, patches):
    vertex_ids = [mesh_vertex(*pt, z, vertices, lookup) for z in [0.0, 0.1] for pt in points]
    blocks.append((vertex_ids, (nx, ny, 1), (gx, gy, 1)))
    sides = [
        (vertex_ids[0], vertex_ids[4], vertex_ids[5], vertex_ids[1]),
        (vertex_ids[1], vertex_ids[5], vertex_ids[6], vertex_ids[2]),
        (vertex_ids[2], vertex_ids[6], vertex_ids[7], vertex_ids[3]),
        (vertex_ids[3], vertex_ids[7], vertex_ids[4], vertex_ids[0]),
    ]
    for side, patch in boundary.items():
        patches[patch].append(sides[side])
    patches['frontAndBack'] += [
        (vertex_ids[3], vertex_ids[2], vertex_ids[1], vertex_ids[0]),
        tuple(vertex_ids[4:]),
    ]

def outer_point(angle):
    a = math.radians(angle)
    return (1 + 10 * math.cos(a), 10 * math.sin(a))

def wake_edge(start, end, wake_cells, trailing_upper, trailing_lower, edges, vertices, lookup):
    local_cells = 6
    ratio = 30 ** (1 / (wake_cells - 1))
    fraction = (ratio ** local_cells - 1) / (ratio ** wake_cells - 1)
    length = (end[0] - start[0]) * fraction
    base_slope = (end[1] - start[1]) / (end[0] - start[0])
    normal_slope = -(trailing_upper[0] - trailing_lower[0]) / (trailing_upper[1] - trailing_lower[1])
    samples = []
    for j in range(81):
        dx = length * j / 80
        offset = (normal_slope - base_slope) * dx * (1 - dx / length) ** 3
        samples.append((start[0] + dx, start[1] + base_slope * dx + offset))
    near_length = sum((math.dist(samples[j], samples[j + 1]) for j in range(80)))
    far_length = math.dist(samples[-1], end)
    near_fraction = near_length / (near_length + far_length)
    for z in [0.0, 0.1]:
        edges.append('polyLine %d %d\n(\n%s\n)' % (
            mesh_vertex(*start, z, vertices, lookup),
            mesh_vertex(*end, z, vertices, lookup),
            '\n'.join(('    (%.17g %.17g %.17g)' % (*sample, z) for sample in samples[1:])),
        ))
    return '((%.17g %d %.17g) (%.17g %d %.17g))' % (
        near_fraction,
        local_cells,
        ratio ** (local_cells - 1),
        1 - near_fraction,
        wake_cells - local_cells,
        ratio ** (wake_cells - local_cells - 1),
    )

def write_mesh_dictionary():
    angle_degrees = 15.0
    rotation = -math.radians(angle_degrees)
    vertices = []
    lookup = {}
    blocks = []
    edges = []
    patches = {n: [] for n in ['airfoil', 'inlet', 'outlet', 'upper', 'lower', 'frontAndBack']}
    leading_edge = profile_point(0, True, rotation)
    trailing_upper = profile_point(1, True, rotation)
    trailing_lower = profile_point(1, False, rotation)
    wake_centre_y = (trailing_upper[1] + trailing_lower[1]) / 2
    chord_positions = [0.0, 0.001, 0.005, 0.02, 0.08, 0.3, 0.6, 1.0]
    lower_counts = [12, 14, 20, 28, 42, 44, 50]
    upper_counts = [14, 18, 24, 32, 48, 44, 50]
    surface_cells = sum(upper_counts) + sum(lower_counts)
    leading_angle = 180 + math.degrees(math.atan(0.2)) - angle_degrees
    upper_angles = [leading_angle, 160, 150, 133, 115, 103, 93, 90]
    lower_angles = [leading_angle, 199 - angle_degrees, 215 - angle_degrees, 227, 245, 257, 267, 270]
    top = (1.0, 10.0)
    bottom = (1.0, -10.0)
    outlet_upper = (25.0, wake_centre_y + 0.003)
    outlet_lower = (25.0, wake_centre_y - 0.003)
    outlet_top = (25.0, 10.0)
    outlet_bottom = (25.0, -10.0)
    profile_cells_per_side = sum(lower_counts)
    normal_cells = 120
    wake_cells = 400
    normal_grading = '((0.012 24 3) (0.138 54 2) (0.85 42 20))'
    reverse_normal_grading = '((0.85 42 0.05) (0.138 54 0.5) (0.012 24 0.3333333333333333))'
    for upper, angles, local_counts in [
        (True, upper_angles, upper_counts),
        (False, lower_angles, lower_counts),
    ]:
        for k, n in enumerate(local_counts):
            x0, x1 = chord_positions[k:k + 2]
            a = profile_point(x0, upper, rotation)
            b = profile_point(x1, upper, rotation)
            c = outer_point(angles[k + 1])
            d = outer_point(angles[k])
            if upper:
                add_mesh_block(
                    [a, b, c, d],
                    n,
                    normal_cells,
                    1,
                    normal_grading,
                    {0: 'airfoil', 2: 'inlet'},
                    vertices,
                    lookup,
                    blocks,
                    patches,
                )
            else:
                add_mesh_block(
                    [b, a, d, c],
                    n,
                    normal_cells,
                    1,
                    normal_grading,
                    {0: 'airfoil', 2: 'inlet'},
                    vertices,
                    lookup,
                    blocks,
                    patches,
                )
            samples = []
            for j in range(1, 160):
                f = j / 160
                x = (math.sqrt(x0) + (math.sqrt(x1) - math.sqrt(x0)) * f) ** 2
                samples.append(profile_point(x, upper, rotation))
            for z in [0.0, 0.1]:
                edges.append('spline %d %d\n(\n%s\n)' % (
                    mesh_vertex(*a, z, vertices, lookup),
                    mesh_vertex(*b, z, vertices, lookup),
                    '\n'.join(('    (%.14g %.14g %.14g)' % (*pt, z) for pt in samples)),
                ))
                mid = outer_point((angles[k] + angles[k + 1]) / 2)
                edges.append('arc %d %d (%.14g %.14g %.14g)' % (
                    mesh_vertex(*c, z, vertices, lookup),
                    mesh_vertex(*d, z, vertices, lookup),
                    *mid,
                    z,
                ))
    upper_wake_grading = wake_edge(
        trailing_upper,
        outlet_upper,
        wake_cells,
        trailing_upper,
        trailing_lower,
        edges,
        vertices,
        lookup,
    )
    lower_wake_grading = wake_edge(
        trailing_lower,
        outlet_lower,
        wake_cells,
        trailing_upper,
        trailing_lower,
        edges,
        vertices,
        lookup,
    )
    add_mesh_block(
        [trailing_upper, outlet_upper, outlet_top, top],
        wake_cells,
        normal_cells,
        30,
        normal_grading,
        {1: 'outlet', 2: 'upper'},
        vertices,
        lookup,
        blocks,
        patches,
    )
    add_mesh_block(
        [bottom, outlet_bottom, outlet_lower, trailing_lower],
        wake_cells,
        normal_cells,
        30,
        reverse_normal_grading,
        {0: 'lower', 1: 'outlet'},
        vertices,
        lookup,
        blocks,
        patches,
    )
    add_mesh_block(
        [trailing_lower, outlet_lower, outlet_upper, trailing_upper],
        wake_cells,
        2,
        30,
        1,
        {1: 'outlet', 3: 'airfoil'},
        vertices,
        lookup,
        blocks,
        patches,
    )
    lines = [
        'FoamFile { format ascii; class dictionary; object blockMeshDict; }',
        'convertToMeters 1;',
        'vertices',
        '(',
    ]
    lines += ['    (%.14g %.14g %.14g)' % pt for pt in vertices]
    lines += [');', 'blocks', '(']
    for index, (vertex_ids, n, g) in enumerate(blocks):
        if index == 14:
            left = normal_grading
            right = '((0.02 24 1) (0.13 54 2) (0.85 42 20))'
        elif index == 15:
            left = reverse_normal_grading
            right = '((0.85 42 0.05) (0.13 54 0.5) (0.02 24 1))'
        if index in [14, 15]:
            if index == 14:
                grading = [
                    upper_wake_grading,
                    30,
                    30,
                    upper_wake_grading,
                    left,
                    right,
                    right,
                    left,
                    1,
                    1,
                    1,
                    1,
                ]
            else:
                grading = [
                    30,
                    lower_wake_grading,
                    lower_wake_grading,
                    30,
                    left,
                    right,
                    right,
                    left,
                    1,
                    1,
                    1,
                    1,
                ]
            method = 'edgeGrading'
        elif index == 16:
            grading = [
                lower_wake_grading,
                upper_wake_grading,
                upper_wake_grading,
                lower_wake_grading,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
            method = 'edgeGrading'
        else:
            grading = g
            method = 'simpleGrading'
        lines.append('    hex (%s) (%s) %s (%s)' % (
            ' '.join(map(str, vertex_ids)),
            ' '.join(map(str, n)),
            method,
            ' '.join(map(str, grading)),
        ))
    lines += [');', 'edges', '(', *edges, ');', 'boundary', '(']
    for name, faces in patches.items():
        if name == 'frontAndBack':
            patch_type = 'empty'
        else:
            patch_type = 'wall' if name == 'airfoil' else 'patch'
        lines += [
            '    ' + name,
            '    {',
            '        type ' + patch_type + ';',
            '        faces',
            '        (',
        ]
        lines += ['            (' + ' '.join(map(str, f)) + ')' for f in faces]
        lines += ['        );', '    }']
    lines += [');', 'mergePatchPairs ();']
    (case / 'system' / 'blockMeshDict').write_text('\n'.join(lines) + '\n')
    (case / 'mesh_design.json').write_text(json.dumps(
        {
            'profile': 'NACA 4412',
            'angle_degrees': angle_degrees,
            'rotation_degrees': -angle_degrees,
            'rotation_centre': [0.25, 0],
            'wake_centre_y': wake_centre_y,
            'chord': 1,
            'thickness_formula_last_coefficient': -0.1015,
            'trailing_edge': 'finite, unmodified',
            'topology': '17-block C-grid',
            'exterior_spacing': 'straight radial blocks with approximately symmetric upper and lower farfield clustering',
            'profile_cells_upper': sum(upper_counts),
            'profile_cells_lower': sum(lower_counts),
            'normal_cells': normal_cells,
            'wake_cells': wake_cells,
            'trailing_edge_cells': 2,
            'wake_reorientation_columns': 6,
            'cells': surface_cells * normal_cells + 2 * wake_cells * normal_cells + wake_cells * 2,
            'normal_grading': [[0.012, 24, 3], [0.138, 54, 2], [0.85, 42, 20]],
            'wake_refined_half_width_near_trailing_edge': 1.5,
            'spanwise_cells': 1,
        },
        indent=2,
    ))
    print(
        'Generated',
        surface_cells * normal_cells + 2 * wake_cells * normal_cells + wake_cells * 2,
        'hexahedra',
    )

def read_foam_list(path):
    text = path.read_text()
    match = re.search('\\n\\s*(\\d+)\\s*\\n\\(\\s*\\n', text)
    if match is None:
        raise ValueError(f'Expected an ASCII OpenFOAM list: {path}')
    return text[match.end():text.rfind(')')]

def read_mesh_edges(mesh):
    text = read_foam_list(mesh / 'points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ').reshape(-1, 3)
    boundary = (mesh / 'boundary').read_text()
    patch = re.search('\\bfrontAndBack\\s*\\{(.*?)\\}', boundary, re.S).group(1)
    first = int(re.search('startFace\\s+(\\d+)', patch).group(1))
    count = int(re.search('nFaces\\s+(\\d+)', patch).group(1))
    lines = read_foam_list(mesh / 'faces').splitlines()[first:first + count]
    faces = re.findall('4\\s*\\(([^)]+)\\)', '\n'.join(lines))
    if len(faces) != count:
        raise ValueError('Expected quadrilateral front and back faces')
    quads = np.fromstring(' '.join(faces), sep=' ', dtype=int).reshape(-1, 4)
    bottom = points[:, 2].min()
    quads = quads[np.all(np.isclose(points[quads, 2], bottom), axis=1)]
    edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return (points[edges, :2], len(quads))

def draw_mesh_panel(ax, segments, bounds, title, width):
    x0, x1, y0, y1 = bounds
    minimum = segments.min(axis=1)
    maximum = segments.max(axis=1)
    visible = (maximum[:, 0] >= x0) & (minimum[:, 0] <= x1)
    visible &= (maximum[:, 1] >= y0) & (minimum[:, 1] <= y1)
    ax.add_collection(LineCollection(segments[visible], colors='#315269', linewidths=width))
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect='equal', xlabel='x/c', ylabel='y/c', title=title)
    ax.spines[['top', 'right']].set_visible(False)

def save_mesh_images(case, segments, count):
    design = json.loads((case / 'mesh_design.json').read_text())
    angle = design['angle_degrees']
    radians = np.deg2rad(-angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    leading = (np.array([0.0, 0.0]) - [0.25, 0.0]) @ rotation.T + [0.25, 0.0]
    trailing = (np.array([1.0, 0.0]) - [0.25, 0.0]) @ rotation.T + [0.25, 0.0]
    plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
    draw_mesh_panel(ax, segments, (-9.5, 25.5, -10.5, 10.5), 'Full domain', 0.15)
    fig.suptitle(f'NACA 4412 | {angle:g} deg | {count:,} cells', fontsize=18)
    fig.savefig(case / 'mesh_overview.png', dpi=200, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 8),
        layout='constrained',
        gridspec_kw={'width_ratios': [1.8, 1, 1]},
    )
    draw_mesh_panel(axes[0], segments, (-0.12, 1.12, -0.38, 0.32), 'Profile', 0.3)
    for ax, centre, title, radius in [
        (axes[1], leading, 'Leading edge', 0.035),
        (axes[2], trailing, 'Trailing edge', 0.045),
    ]:
        x, y = centre
        draw_mesh_panel(ax, segments, (x - radius, x + radius, y - radius, y + radius), title, 0.45)
    fig.suptitle(f'NACA 4412 | {angle:g} deg | mesh detail', fontsize=18)
    fig.savefig(case / 'mesh_detail.png', dpi=200, facecolor='white')
    plt.close(fig)
    print(case / 'mesh_overview.png')
    print(case / 'mesh_detail.png')

def run(command, log_name):
    print('+ ' + ' '.join(command), flush=True)
    log_path = case / log_name
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit('Command failed. Read ' + str(log_path))

def generate_mesh():
    write_mesh_dictionary()
    run(['blockMesh'], 'log.blockMesh')
    run(['checkMesh', '-allGeometry', '-allTopology'], 'log.checkMesh')
    if 'Mesh OK.' not in (case / 'log.checkMesh').read_text():
        raise SystemExit('Mesh checks failed. Read ' + str(case / 'log.checkMesh'))
    print('Mesh OK.', flush=True)

def plot_mesh():
    segments, cell_count = read_mesh_edges(case / 'constant' / 'polyMesh')
    save_mesh_images(case, segments, cell_count)

def simulate():
    decomposition = (case / 'system/decomposeParDict').read_text()
    match = re.search(r'numberOfSubdomains\s+(\d+)\s*;', decomposition)
    processes = match.group(1)
    run(['decomposePar'], 'log.decomposePar')
    run(['mpirun', '-np', processes, 'foamRun', '-parallel'], 'log.foamRun')
    run(['reconstructPar', '-newTimes'], 'log.reconstructPar')
    print('Simulation finished; fields reconstructed.', flush=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate the mesh, plot it and run the NACA 4412 case in the background.')
    parser.add_argument(
        '--mesh-only',
        action='store_true',
        help='Generate, check and plot the mesh without running the solver.',
    )
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()

def check_commands(mesh_only):
    commands = ['blockMesh', 'checkMesh']
    if not mesh_only:
        commands.extend(['decomposePar', 'mpirun', 'foamRun', 'reconstructPar'])
    for command in commands:
        if shutil.which(command) is None:
            raise SystemExit('Command unavailable: ' + command + '. Run: source /opt/openfoam14/etc/bashrc')

def check_existing_results():
    for path in case.iterdir():
        if path.name.startswith('processor') or path.name in ['postProcessing', 'log.foamRun']:
            raise SystemExit('Existing simulation found: ' + str(path) + '. Keep or remove its results before starting again.')
        if path.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?', path.name):
            if float(path.name) != 0:
                raise SystemExit('Existing simulation found: ' + str(path) + '. Keep or remove its results before starting again.')

def start_background(mesh_only):
    with (case / 'log.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This case is already running. Read ' + str(case / 'log.run'))

        check_existing_results()
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker']
        if mesh_only:
            command.append('--mesh-only')
        with (case / 'log.run').open('w') as log:
            process = subprocess.Popen(
                command,
                cwd=case,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )

    print('Started in background. PID: ' + str(process.pid))
    print('Workflow: tail -f ' + shlex.quote(str(case / 'log.run')))
    if not mesh_only:
        print('Solver:   tail -F ' + shlex.quote(str(case / 'log.foamRun')))

def main():
    arguments = parse_arguments()
    if not arguments.worker:
        check_commands(arguments.mesh_only)
        start_background(arguments.mesh_only)
        return

    generate_mesh()
    plot_mesh()
    if arguments.mesh_only:
        print('Mesh generation finished.', flush=True)
        return
    simulate()

if __name__ == '__main__':
    main()
