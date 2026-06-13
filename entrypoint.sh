#!/bin/bash
set -e

HOST_UID=${HOST_UID:-$(stat -c '%u' /workspace 2>/dev/null || echo 0)}
HOST_GID=${HOST_GID:-$(stat -c '%g' /workspace 2>/dev/null || echo 0)}

# Если уже не root или UID=0 — выполняем напрямую
if [ "$(id -u)" != "0" ] || [ "${HOST_UID}" = "0" ]; then
    exec "$@"
fi

# Находим gosu
GOSU=$(command -v gosu \
    || command -v /usr/sbin/gosu \
    || command -v /usr/local/bin/gosu \
    || echo "")

if [ -z "$GOSU" ]; then
    echo "ПРЕДУПРЕЖДЕНИЕ: gosu не найден, запуск от root"
    exec "$@"
fi

# Создаём группу
if ! getent group "$HOST_GID" > /dev/null 2>&1; then
    groupadd -g "$HOST_GID" hostgroup
fi

# Создаём пользователя
if ! getent passwd "$HOST_UID" > /dev/null 2>&1; then
    useradd -u "$HOST_UID" \
            -g "$HOST_GID" \
            -m \
            -s /bin/bash \
            -d /home/agamauser \
            agamauser
    USERNAME="agamauser"
else
    USERNAME=$(getent passwd "$HOST_UID" | cut -d: -f1)
fi

HOME_DIR=$(getent passwd "$HOST_UID" | cut -d: -f6)
echo "Запуск от имени: ${USERNAME} (UID=${HOST_UID}, GID=${HOST_GID})"
echo "HOME: ${HOME_DIR}"

# Устанавливаем RCLONE_CONFIG явно — не зависит от имени пользователя
# Приоритет: переменная уже задана → оставляем, иначе ставим путь по умолчанию
export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.config/rclone/rclone.conf}"
echo "RCLONE_CONFIG: ${RCLONE_CONFIG}"

exec $GOSU "$HOST_UID:$HOST_GID" "$@"
