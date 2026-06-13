import sys

packages = [
    ('requests',  'requests'),
    ('numpy',     'numpy'),
    ('scipy',     'scipy'),
    ('torch',     'torch'),
    ('botorch',   'botorch'),
    ('gpytorch',  'gpytorch'),
    ('sklearn',   'scikit-learn'),
    ('agama',     'agama'),
    ('cvxopt',    'cvxopt'),
    ('mgefit',    'mgefit'),
    ('powerbin',  'powerbin'),
]

failed = []
for module, name in packages:
    try:
        m = __import__(module)
        ver = getattr(m, '__version__', 'n/a')
        print(f'  OK {name}: {ver}')
    except ImportError as e:
        print(f'  FAIL {name}: {e}')
        failed.append(name)

if failed:
    print(f'\nНе установлены: {failed}')
    sys.exit(1)
else:
    print('\nВсе пакеты установлены успешно')
