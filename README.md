# CaTMU

### Convolución Acelerada por la Unidad de Mapeo de Texturas

##### Requisitos:

  * Dispositivo compatible con CUDA 9.1 o superior
  * Compilador de CUDA nvcc en versión 9.1 o superior con ruta incluida en el sistema.

##### Compilación:

La biblioteca puede ser compilada ejecutando `make all` en la subcarpeta `catmu/cuda`.
Al utilizar el módulo se verifica que la compilación exista y, de ser necesario se 
realiza automáticamente, pero para afectar cambios se debe recompilar manualmente.
  

##### Estructura de carpetas (incompleto):

  * `bin` contiene ejemplos de uso y pruebas

  * `catmu` contiene el módulo
  
  * `catmu/cuda` contiene lo necesario para compilar la biblioteca
