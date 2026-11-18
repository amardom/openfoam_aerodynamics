from pathlib import Path
import argparse
import fcntl
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
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect='equal', xlabel='x/H', ylabel='y/H', title=title)
    ax.spines[['top', 'right']].set_visible(False)

def run(command, log_name):
    print('+ ' + ' '.join(command), flush=True)
    log_path = case / log_name
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit('Command failed. Read ' + str(log_path))


def generate_mesh():
    run(['blockMesh'], 'log.blockMesh')
    run(['checkMesh', '-allGeometry', '-allTopology'], 'log.checkMesh')
    if 'Mesh OK.' not in (case / 'log.checkMesh').read_text():
        raise SystemExit('Mesh checks failed. Read ' + str(case / 'log.checkMesh'))
    print('Mesh OK.', flush=True)

def plot_mesh():
    segments, count = read_mesh_edges(case / 'constant' / 'polyMesh')
    plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
    draw_mesh_panel(ax, segments, (-12.5, 25.5, -12.5, 12.5), 'Full domain', 0.15)
    fig.suptitle(f'Square at 0° | Re=200 | {count:,} cells', fontsize=18)
    fig.savefig(case / 'mesh_overview.png', dpi=200, facecolor='white')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), layout='constrained')
    draw_mesh_panel(axes[0], segments, (-1.6, 1.6, -1.6, 1.6), 'Square at 0°', 0.3)
    draw_mesh_panel(axes[1], segments, (0.40, 0.60, 0.40, 0.60), 'Upper rear corner', 0.45)
    fig.suptitle('Square at 0° | mesh detail', fontsize=18)
    fig.savefig(case / 'mesh_detail.png', dpi=200, facecolor='white')
    plt.close(fig)
    print(case / 'mesh_overview.png')
    print(case / 'mesh_detail.png')

def simulate():
    decomposition = (case / 'system/decomposeParDict').read_text()
    match = re.search(r'numberOfSubdomains\s+(\d+)\s*;', decomposition)
    processes = match.group(1)
    run(['decomposePar'], 'log.decomposePar')
    run(['mpirun', '-np', processes, 'foamRun', '-parallel'], 'log.foamRun')
    run(['reconstructPar', '-newTimes'], 'log.reconstructPar')
    print('Simulation finished; fields reconstructed.', flush=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate the mesh, plot it and run the square at 0 degrees in the background.')
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
