import agama
import numpy

print('OK agama version:', agama.__version__)

# Проверяем что загружен полный пакет (C++ + Python)
assert hasattr(agama, 'nonuniformGrid'), \
    "FAIL: nonuniformGrid недоступен — загружен только C++ модуль!"
print('OK: nonuniformGrid доступен')

# Тест nonuniformGrid
g = agama.nonuniformGrid(nnodes=10, xmin=0.01, xmax=5.0)
assert len(g) == 10, "FAIL: неверная длина сетки"
print('OK: nonuniformGrid тест:', g[:3])

# Тест потенциала
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='Plummer', mass=1.0, scaleRadius=1.0)
val = pot.potential([1.0, 0.0, 0.0])
assert val < 0, "FAIL: потенциал должен быть отрицательным"
print('OK: Plummer потенциал:', val)

# Проверяем schwarzlib
assert hasattr(agama, 'schwarzlib'), \
    "FAIL: schwarzlib недоступен!"
print('OK: schwarzlib доступен')

print('\nВсе проверки AGAMA пройдены успешно')
