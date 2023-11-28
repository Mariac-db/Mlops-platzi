# Configuración de Pyenv

Este es un ejemplo de cómo configurar Pyenv en tu sistema.

## Instalación de Pyenv

Para instalar Pyenv, sigue las instrucciones en la documentación oficial: https://github.com/pyenv/pyenv#installation

## Integración del shell

Para habilitar la integración del shell, abre tu archivo de configuración de shell (~/.zprofile para sesiones de inicio o ~/.zshrc para sesiones interactivas) y agrega al final del archivo las siguientes líneas proporcionadas por Pyenv:

1. Abre tu archivo de configuración de Zsh, que podría ser ~/.zprofile para sesiones de inicio o ~/.zshrc para sesiones interactivas. Puedes usar un editor de texto como Nano o Vim para esto.

nano ~/.zprofile

o 

nano ~/.zshrc

2. Agrega las siguientes líneas al final del archivo:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
``` bash

3. Para que los cambios tengan efecto, reinicia tu shell. Puedes cerrar la terminal y abrirla de nuevo, o ejecutar el comando

source ~/.zprofile

o

source ~/.zshrc


4. Una vez hecho esto, vuelve a intentar seleccionar la versión de Python:

pyenv shell 3.9.17

Y verifica la versión seleccionada de Python:

python --version

5. Crear entorno de trabajo con poetry a partir de la versión específica ed Python, 

poetry env use 3.9.17

poetry new nombre_del_proyecto

mv nombre_del_proyecto/pyproject.toml root_del_proyecto/






