import os
from pathlib import Path, PosixPath
import time
from subprocess import Popen, PIPE
import pickle
from ricetypes import Result, Enum, Scalar_Variant

db_path = Path('/home/mgisborne/Simulations/db/main.db')
assert db_path.exists()
# from BigDFT import InputActions as IA


def sh(proc, *args):
    return Popen([proc, *[str(a) for a in args]], stdout=PIPE)\
            .communicate()[0]\
            .decode('utf-8')\
            .split('\n')


def underline(string):
    return string + '\n' + len(string) * '-'


def align(*lines, sep=' = ', end=''):
    left_pad = max(map(lambda line: len(str(line[0])), lines))
    for left, right in lines:
        left, right = str(left), str(right)
        start = ' ' * (left_pad - len(left))
        yield start + left + sep + right + end


def print_peramiters(title, *lines):
    print()
    print(underline(title))
    print(*align(*lines), sep='\n')
    print()


def mk_slurm_file(dir: Path,
                  jobname: str,
                  time: str,
                  queue: str,
                  nodes: int,
                  tasks: int = None,
                  OMP: int = 1,
                  CPU: int = 32,
                  comments=[],
                  ):

    assert dir.exists(), dir

    tasks = tasks or CPU*nodes//OMP
    bindir = os.getenv('bindir')
    bindir = Path(bindir) if bindir else Path.cwd().joinpath('bin')
    assert bindir.exists(), bindir

    local_bindir = dir.joinpath('bin')
    sh('cp', '-rf', bindir, local_bindir)

    module_file = local_bindir.joinpath("env.sh")
    assert module_file.exists(), module_file

    output = dir.joinpath('out.txt')

    contents = ['#!/bin/bash']
    contents.append('## -- This Script Was Produced Automaticaly -- ##')
    for comment in comments:
        contents.append('# ' + comment)

    contents.extend(2*[''])

    contents.append('#SBATCH -J ' + str(jobname))
    contents.append('#SBATCH -o ' + str(output))
    contents.append('#SBATCH -N ' + str(nodes))
    contents.append('#SBATCH -n ' + str(tasks))
    contents.append('#SBATCH -t ' + str(time))
    contents.append('#SBATCH -p ' + str(queue))
    contents.extend(2*[''])

    contents.append('echo starting at $(date)')
    contents.append(f'cd {dir}')
    contents.append(f'source {module_file.absolute()}')
    contents.append(f'PATH={local_bindir.absolute()}:$PATH')
    contents.append(f'export OMP_NUM_THREADS={OMP}')
    contents.extend(2*[''])

    contents.append('# Change job status to running')
    contents.append('runid=$(cat run_rowid)')
    contents.append(f'''echo "update runs set status='R' where rowid=$runid;" | sqlite3 {db_path}''')
    contents.extend(2*[''])

    contents.append('mpirun --ppn 1 profiler.py &')
    contents.append('runcode.py')
    contents.append('touch done')
    contents.append('kill %%')
    contents.extend(2*[''])

    contents.append('# Change job status to Complete')
    contents.append(f'''echo "update runs set status='C' where rowid=$runid;" | sqlite3 {db_path}''')
    contents.extend(2*[''])

    contents.append('echo ending at $(date)')

    slurm_file = dir.joinpath('slurm.sh')
    with open(slurm_file, 'w') as file:
        file.write('\n'.join(contents))

    sh('chmod', '+x', slurm_file)
    return slurm_file


def formatname(**args):
    'formats name in key[args]-key[args] format.'
    return '-'.join([f'{k}[{v}]' for k, v in args.items() if v is not None])


def processname(name: str) -> dict:
    data = dict()
    for info in name.split('-'):
        key, value, _ = info.split('[')
        data[key] = value
    return data


Requied = object()
Default = object()


class Callback:
    def __init__(self, callback):
        self.callback = callback


@Enum
class yes_or_no:
    yes: Scalar_Variant
    no: Scalar_Variant


@Enum
class Queue_Type:
    normal: Scalar_Variant
    test: Scalar_Variant


@Enum
class System_Types:
    PGP: Scalar_Variant
    PGP_Mel: Scalar_Variant

    def to_string(self):
        match self:
            case System_Types.PGP:
                return 'PGP'
            case System_Types.PGP_Mel:
                return 'PGP-Mel'


@Enum
class XC_Type:
    PBE: Scalar_Variant

config_type = {'env': {'OMP': (int, 4),
                       'nodes': (int, Requied),
                       'root': ({str, PosixPath}, Requied),
                       'queue': (Queue_Type, Queue_Type.normal),
                       },
               'system': {
                   'name': (System_Types, Requied),
                   'atoms': (int, Requied),
                   'posinp': {'positions': (list, Requied),
                              'units': (str, Requied),
                              'cell': (list, Requied),
                              #'properties': {'reduced': (Enum('yes', 'no'), 'no')},
                              },
               },
               'space': {
                   'h': (float, .4),
                   'f': (float, 4.5),
                   'c': (float, 6.5),
               },
               'algorithm': {
                   'configerations_id': (int, Requied),
                   'linear': (bool, True),
                   'xc': (XC_Type, XC_Type.PBE),
                   'convergence': {'gnrm':     (Callback(lambda v: (type(v) is float) or (v == 'default')), .01),
                                   'rpnrm': (Callback(lambda v: (type(v) is float) or (v == 'default')), 'default')},
               },
               'post': {
                   'write_support_functions': (bool, True)
               },
               'name': ({str, type(None)}, None),
               }


def merge(A: dict, B: dict) -> Result:  # Result<Dict, Tuple[*keys, message]>
    keys = set(A) | set(B)
    out = {}
    for key in keys:
        if (type(A.get(key)) is dict) or (type(B.get(key)) is dict):
            if key not in A:
                recursion = merge({}, B[key])
            elif key not in B:
                recursion = merge(A[key], {})
            elif (type(A.get(key)) is dict) and (type(B.get(key)) is dict):
                recursion = merge(A[key], B[key])
            else:
                return Result.Error((key, f'If either A or B are dicts, then both must be dicts. but A:{type(A[key])}, B:{type(B[key])}'))

            if recursion.error:
                return Result.Error((key, *recursion._error))

            out[key] = recursion.unwrap()
            continue

        if key in B:
            out[key] = B[key]
        else:
            out[key] = A[key]

    return Result.Ok(out)


Left = merge({'a': {'b': 1, 'c': 2}}, {'a': {'b': 3}}).unwrap()
Right = {'a': {'b': 3, 'c': 2}}
assert merge({'a': {'b': 1, 'c': 2}}, {'a': {'b': 3}}).unwrap() == {'a': {'b': 3, 'c': 2}}, (Left, Right)
assert merge({'a': {'b': 3}}, {'a': {'b': 1, 'c': 2}}).unwrap() == {'a': {'b': 1, 'c': 2}}
assert merge({'a': {'b': 1}}, {'a': 2}).error
assert merge({'a': {'b': 1}}, {'a': 2}).error


def mkconfig(inp: dict, types=config_type, check=True):
    if not set(inp).issubset(set(types)):
        return Result.Error(('Unknown Keys', set(inp) - set(types)))

    out = dict()
    for key in types:
        if type(types[key]) is dict:
            if key not in inp:
                res = mkconfig({}, types[key], check=check)

            elif type(inp[key]) is not dict:
                return Result.Error((key, 'value should be dict'))

            else:
                res = mkconfig(inp[key], types[key], check=check)

            if res.error:
                return Result.Error((key, *res._error))

            out[key] = res.unwrap()

            continue

        t, default = types[key]
        if not check:
            out[key] = inp.get(key, default)
            continue

        if key not in inp and default is Requied:
            return Result.Error((key, f'Requied Key Missing of type {t}'))

        if key not in inp or inp[key] is Default:
            val = default
        else:
            val = inp[key]

        if type(t) is set:
            if type(val) not in t:
                return Result.Error((key, f'Type Error, Exspecting one of {t}, found {type(val)}, value given = {val}'))
        elif type(t) is Enum:
            if val not in t:
                return Result.Error((key, f'Type Error, Expecting a {t}, but {val} is not a valid variant'))
        elif type(t) is Callback:
            if not t.callback(val):
                return Result.Error((key, f'Type Error. Found given {type(val)}, value given = {val}. Callable type check failed')) 
        else:
            if type(val) is not t:
                return Result.Error((key, f'Type Error, Exspecting {t}, found {type(val)}, value given = {val}'))

        out[key] = val

    return Result.Ok(out)


def mkinput(config):
    config = mkconfig(config).unwrap()
    inp = {
        'dft': {'rmult': [config['space']['f'], config['space']['c']],
                'hgrids': config['space']['h'],
                'ixc': str(config['algorithm']['xc']),
                'gnrm_cv': config['algorithm']['convergence']['gnrm'],
                },
        'lin_general': {'rpnrm_cv': config['algorithm']['convergence']['rpnrm'],
                        'subspace_diag':  config['algorithm']['linear'],
                        },
        'mix': {'rpnrm_cv': config['algorithm']['convergence']['rpnrm']},
    }

    if config['post']['write_support_functions']:
        # AI.write_support_function_matrices()
        inp['lin_general']['output_mat'] = 1  # 1 for text, 4 for binary

    if config['algorithm']['linear']:
        inp['import'] = 'linear'
        #inp.setdefault("lin_general", {}).update({'subspace_diag': True})

    return inp


def create_simulation(config: dict) -> Path:
    config = mkconfig(config).unwrap()
    input = mkinput(config)

    name = str(config.get('name')) or (
            formatname(OMP=config['env']['OMP'],
                       atoms=config['system']['atoms'],
                       # rmult=(config['space']['f'], config['space']['c']),
                       hgrid=config['space']['h'],
                       linear=config['algorithm']['linear'],
                       nodes=config['env']['nodes'],
                       xc=config['algorithm']['xc'],
                       ))

    config['name'] = name
    root = config['env']['root']
    dir = root.joinpath(name)
    dir.mkdir(parents=True, exist_ok=False)

    mk_slurm_file(dir=dir,
                  jobname=name,
                  nodes=config['env']['nodes'],
                  time='72:00:00',
                  OMP=config['env']['OMP'],
                  queue=config['env']['queue'],
                  comments=[f'name: {name}',
                            f'created: {time.ctime()}'],)

    args = dict(input=input,
                posinp=config['system']['posinp'],
                cell=config['system']['posinp']['cell'],
                name='',
                run_dir='run_dir'
                )

    with open(dir.joinpath("args.pickle"), "wb") as f:
        pickle.dump(args, f)

    return dir


def query_db(query_string: str, peramiters=(), dry_run=True):
    import sqlite3
    local_db_path = db_path
    assert local_db_path.exists(), local_db_path
    if dry_run:
        from tempfile import mktemp
        temp_db_path = Path(mktemp()).with_suffix('.db')
        print(f'creating temp db for dry run at: {temp_db_path}')
        sh('cp', '-v', local_db_path, temp_db_path)
        local_db_path = temp_db_path

    try:
        db = sqlite3.connect(local_db_path)

        class Row(sqlite3.Row):
            def __repr__(self):
                return f"Row({', '.join([f'{key}={self[key]}' for key in self.keys() ])})"

        db.row_factory = Row
        with db as cursor:
            cursor = cursor.execute(query_string, peramiters)
            while True:
                results = cursor.fetchmany(5)
                if len(results) == 0:
                    return
                yield from results

    except sqlite3.Error as error:
        print("Sqlite3 Error:", error, query_string, peramiters)

    finally:
        if dry_run:
            print('removing temp db')
            sh('rm', '-v', temp_db_path)


XC_lookup = {
    101: 'PBE',
}


def config_row_to_config(row):
    return {'space': {'h': row['hgrid'], 'f': row['rmult_f'], 'c': row['rmult_c']},
            'algorithm': {'convergence': {'gnrm': row['convg']},
                          'linear': True if row['linear'] == 1 else False,
                          'xc': XC_lookup[row['xc']],
                          }
            }


class Simulation:
    config_checked = False
    dir_written = False
    added_to_db = False
    path = None
    rowid = None

    def __init__(self, config):
        self.config = mkconfig(config, check=False).unwrap()

    def set_configeration(self, config_id: int):
        config = list(query_db('''select rowid,* from configerations where rowid=?;''', (config_id,), dry_run=False))

        if len(config) != 1:
            raise Exception('There should be one and only one configeration to each config_id', config_id, config)

        return (Result.Ok(self.config)
                .bind(merge, config_row_to_config(config[0]))
                .bind(merge, {'algorithm': {'configerations_id': config_id}})
                .unwrap())

    def check_config(self):
        self.config = mkconfig(self.config).unwrap()
        self.config_checked = True
        return self

    def write_dir(self):
        assert self.config_checked
        path = create_simulation(self.config)
        self.dir_written = True
        self.path = path
        return self

    def add_to_db(self):
        assert self.dir_written
        path = self.path
        config = self.config
        list(query_db('''insert into runs(system,   config, enviroment, atoms,  nodes, OMP,   MPI,  date,        status,  path)
                                     values (:system,       :config, 1        , :atoms, :nodes, :OMP, :MPI, date(:date), :status, :path)''',
                         dict(
                             system=config['system']['name'],
                             config=config['algorithm']['configerations_id'],
                             atoms=config['system']['atoms'],
                             nodes=config['env']['nodes'],
                             OMP=config['env']['OMP'],
                             MPI=32//config['env']['OMP'],
                             date=time.strftime('%Y-%m-%d'),
                             status='N',
                             path=str(path.absolute())
                         ),
             dry_run=False))

        self.rowid = list(query_db('select rowid from runs order by rowid desc limit 1;', dry_run=False))[0]['rowid']

        with open(path.joinpath('run_rowid'), 'w') as f:
            f.write(f'{self.rowid}')

        self.added_to_db = True

        return self
