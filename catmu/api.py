import ctypes
import logging
import pathlib
import threading
from typing import Tuple, Iterable, Union

import numpy as np

from catmu.structures import (Image, Positions, LutPSF, ExpressionPSF,
                              DevConfig, DeviceProperties)

# Configuración del módulo Logging
logging.basicConfig(format="%(threadName)s : %(message)s", level='INFO')
logger = logging.getLogger(__name__)

# Punteros a cada tipo de estructura
pImage = ctypes.POINTER(Image)
pPositions = ctypes.POINTER(Positions)
pLutPSF = ctypes.POINTER(LutPSF)
pExprPSF = ctypes.POINTER(ExpressionPSF)
pConfig = ctypes.POINTER(DevConfig)
pDeviceProperties = ctypes.POINTER(DeviceProperties)
callbackType = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_double, ctypes.c_ulong)

# Definiciones de tipos
Size2D = Tuple[int, int]
Size3D = Tuple[int, int, int]
PixelSize2D = Tuple[float, float]
PixelSize3D = Tuple[float, float, float]
ValidSizes = Union[Size2D, Size3D]
ValidPixelSizes = Union[PixelSize2D, PixelSize3D]


def compile(lib_name: str = "libConvolve.so"):  # pragma: no cover
    print(f"Compilando {lib_name}...")
    import subprocess
    make_command = f'make -C {pathlib.Path(__file__).parent / "cuda"} all'
    subprocess.run(make_command, shell=True)
    lib_path = pathlib.Path(pathlib.Path(__file__).parent / f'cuda/{lib_name}')
    if lib_path.exists() is False:
        raise FileNotFoundError(f'La biblioteca {lib_path} no pudo ser compilada.')
    print("Compilación exitosa")


def load_library(debug=False) -> ctypes.CDLL:
    # Selecciona la biblioteca a utilizar (modo debug o no)
    if debug is True:
        lib_name = 'libConvolveDebug.so'
    else:
        lib_name = 'libConvolve.so'

    lib_path = pathlib.Path(pathlib.Path(__file__).parent / f'cuda/{lib_name}')

    # Si la biblioteca no existe intenta compilarla
    if lib_path.exists() is False:  # pragma: no cover
        compile(lib_name)

    # Carga la biblioteca
    lib = ctypes.CDLL(lib_path)

    # Configura los tipos recibidos y el tipo devuelto
    lib.getDevProp.argtypes = [ctypes.c_int, pDeviceProperties]
    lib.getDevProp.restype = ctypes.c_int

    lib.lutConvolution.argtypes = [callbackType, pImage, pPositions, pLutPSF, pConfig]
    lib.exprConvolution2D.argtypes = [callbackType, pImage, pPositions, pExprPSF, pConfig]
    lib.lutConvolution.restype = ctypes.c_int
    lib.exprConvolution2D.restype = ctypes.c_int

    lib.cpu_lut_convolve2D.argtypes = [pImage, pPositions, pLutPSF]
    lib.cpu_lut_convolve2D.restype = ctypes.c_int
    lib.cpu_lut_convolve2D_openmp.argtypes = [pImage, pPositions, pLutPSF]
    lib.cpu_lut_convolve2D_openmp.restype = ctypes.c_int

    lib.cpu_expr_convolve2D.argtypes = [pImage, pPositions, pExprPSF]
    lib.cpu_expr_convolve2D.restype = ctypes.c_int
    lib.cpu_expr_convolve2D_openmp.argtypes = [pImage, pPositions, pExprPSF]
    lib.cpu_expr_convolve2D_openmp.restype = ctypes.c_int

    return lib


lib = load_library(debug=False)
lib_debug = load_library(debug=True)


def get_available_devices() -> int:
    count = ctypes.c_int(0)
    r = lib.get_available_devices(ctypes.byref(count))
    if r != 0:  # pragma: no cover
        raise CatmuError(code=r)
    return int(count.value)


def get_device_properties(dev: int = 0) -> DeviceProperties:
    devProp = DeviceProperties()
    r = lib.getDevProp(dev, devProp)
    if r != 0:  # pragma: no cover
        raise CatmuError(code=r)
    return devProp


class CatmuError(Exception):  # pragma: no cover
    """ Excepción interna de la clase ConvolutionManager

    Permite capturar los posibles errores internos del módulo.
    """

    def __init__(self, message=None, code=None):

        self.code = code

        if code == 2:
            self.cuda_message = ("El dispositivo no tiene suficiente memoria "
                                 "para completar la operación")
        elif code == 100:
            self.cuda_message = ("No se detectó ningún dispositivo con capacidad "
                                 "para ejecutar CUDA (posible falla del driver).")
        elif code == 101:
            self.cuda_message = ("El número de dispositivo requerido no "
                                 "corresponde a un abonado al servicio.")
        elif code == 999:
            self.cuda_message = ("Ocurrió un error interno de CUDA desconocido.")
        elif code is None:
            self.cuda_message = ""
        else:
            self.cuda_message = (f"Código de error {code}. Consultar la "
                                 f"documentación de CUDA.")

        if message is None:
            if code is not None:
                self.message = f'CudaError ({code}): {self.cuda_message}'
            else:
                self.message = 'Mensaje de error no implementado'
        else:
            self.message = message


class ConvolutionManager:
    def __init__(self, debug: bool = False):
        # Carga la libreria
        if debug is False:   # pragma: no cover
            self._lib = lib
        else:
            self._lib = lib_debug

        self._device = 0

        self._dim = None

        # Arrays de Numpy con la memoria local
        self._psf = None
        self._id_function = None
        self._params = None
        self._positions = None
        self._results = None

        self._s_psf = None
        self._s_config = None
        self._s_positions = None
        self._s_image = None

        # Configuraciones de tamaños de píxel
        self._image_size = None
        self._psf_pixel_size = None
        self._image_pixel_size = None

        # Eventos de señalización
        self._daemon_running = threading.Event()
        self._daemon_ready = threading.Event()

        # Thread secundario
        self._daemon_thread = None

        # Timeout
        self._timeout = None

    @property
    def active(self):   # pragma: no cover
        return True

    def prepare_lut_psf(self, psf: np.ndarray,
                        image_size: ValidSizes = (64, 64),
                        image_pixel_size: ValidPixelSizes = (1.0, 1.0),
                        psf_pixel_size: ValidPixelSizes = (1.0, 1.0)):

        """ Configura una sesión de trabajo con una PSF de tipo LUT.

        Este método pone a correr un hilo secundario que hace de interfaz con CUDA.
        El vínculo con la GPU se mantiene activo a través de ese hilo sin detener
        la ejecución principal. Cuando se solicita una convolución, el hilo principal
        se detiene y espera hasta que el hilo secundario obtenga los resultados de la GPU.

        La PSF se carga una sola vez al configurar la sesión. Para cambiar la PSF se debe
        llamar este método nuevamente, provocando que el hilo secundario existente termine
        y se de comienzo a uno nuevo.

        Parameters
        ----------
        psf : np.ndarray
            PSF a cargar en memoria.
        image_size : Tuple[int, int]
            Tamaño de la imagen resultante
        image_pixel_size : Tuple[float, float]
            Tamaño del pixel de la imagen
        psf_pixel_size : Tuple[float, float]
            Tamaño del pixel de la psf

        """

        self._background_shutdown()

        # Configura la PSF
        if not isinstance(psf, (tuple, list, np.ndarray)):   # pragma: no cover
            raise TypeError

        self._dim = psf.ndim

        if len(image_size) != self._dim:   # pragma: no cover
            raise ValueError(f'PSF DIM: {self._dim} != IMAGE DIM: {len(image_size)}')

        self._psf = np.asarray(psf, dtype=ctypes.c_float, order='c')

        # Configura el tamaño de la imagen resultante y los tamaños de píxeles
        self._image_size = image_size
        self._image_pixel_size = image_pixel_size
        self._psf_pixel_size = psf_pixel_size

        self._s_psf = LutPSF(psf_data=self._psf,
                             pixel_size=self._psf_pixel_size)

        self._s_image = Image()
        self._s_positions = Positions()

        self._prepare()

    def prepare_expression_psf(self, id_function: int, params: Iterable,
                               image_size: ValidSizes = (64, 64),
                               image_pixel_size: ValidPixelSizes = (1.0, 1.0)):
        """ Configura una sesión de trabajo con una PSF de tipo Expresión.

        Este método pone a correr un hilo secundario que hace de interfaz con CUDA.
        El vínculo con la GPU se mantiene activo a través de ese hilo sin detener
        la ejecución principal. Cuando se solicita una convolución, el hilo principal
        se detiene y espera hasta que el hilo secundario obtenga los resultados.

        La PSF se carga una sola vez al configurar la sesión. Para cambiar la PSF se debe
        llamar este método nuevamente, provocando que el hilo secundario existente termine
        y se de comienzo a uno nuevo.

        Parameters
        ----------
        psf : np.ndarray
            PSF a cargar en memoria.
        image_size : Tuple[int, int]
            Tamaño de la imagen resultante
        image_pixel_size : Tuple[float, float]
            Tamaño del pixel de la imagen
        psf_pixel_size : Tuple[float, float]
            Tamaño del pixel de la psf

        """

        self._dim = 2

        self._background_shutdown()

        # Configura la PSF
        self._id_function = id_function
        self._params = np.asarray(params)

        # Configura el tamaño de la imagen resultante y los tamaños de píxeles
        self._image_size = image_size
        self._image_pixel_size = image_pixel_size

        self._s_psf = ExpressionPSF(id_function=self._id_function,
                                    params=self._params)

        self._s_image = Image()
        self._s_positions = Positions()

        self._prepare()

    def async_convolve(self, positions: np.ndarray):
        """ Método que comienza el proceso de convolución en forma asincrónica

        Este método da la señal al hilo secundario para que comience la tarea de
        convolucionar la lista de posiciones indicadas.

        La función devuelve el control inmediatamente y deja a cargo del usuario la espera
        de los resultados.

        El método get_results marca la barrera, requiriendo que la convolución finalice
        para continuar con la ejecución del hilo principal.

        Para convoluciones sincronicas ver: sync_convolve

        Parameters
        ----------
        positions : np.ndarray
            Lista de posiciones con formato [N, n, 2/3] donde N representa la cantidad de
            "individuos", n la cantidad de fuentes virtuales y 2/3 la dimensión permitida.

        """

        # Configura la lista de posiciones
        if isinstance(positions, (tuple, list, np.ndarray)):
            self._positions = np.asarray(positions, dtype=ctypes.c_float, order='c')
        else:
            raise TypeError   # pragma: no cover

        n = len(self._positions)

        if self._dim == 2:
            result_dim = (
                n,
                self._image_size[0],
                self._image_size[1]
            )
        elif self._dim == 3:
            result_dim = (
                n,
                self._image_size[0],
                self._image_size[1],
                self._image_size[2]
            )
        else:
            raise RuntimeError   # pragma: no cover

        self._results = np.zeros(result_dim, dtype=ctypes.c_float, order='c')

        self._daemon_ready.clear()

        self._start_convolution()

    def sync_get_results(self, get_copy: bool = True) -> np.ndarray:
        """ Método para esperar resultados generados por async_convolve

        Este método sirve como barrera para detener la ejecución del hilo principal hasta
        que la convolución haya finalizado y además devuelve los resultados obtenidos.

        Para convoluciones sincrónicas ver: sync_convolve

        Parameters
        ----------
        get_copy : bool
            Indica si el resultado devuelto es una copia de la memoria utilizada por la
            interfaz. Las copias son más seguras, pero también un poco más lentas.
            Para operaciones de sólo lectura en forma segura la copia no es necesaria
            (por ejemplo para calcular una métrica sobre la convolución devuelta sin
            afectar sus valores).

        Returns
        -------
        Las imágenes resultantes de la convolución o una copia de ellas
        (según el parámetro get_copy).

        """
        if self._daemon_ready.wait(timeout=self._timeout) is False:
            raise TimeoutError   # pragma: no cover

        if get_copy is True:
            return np.copy(self._results)
        else:   # pragma: no cover
            return self._results

    def sync_convolve(self, positions: np.ndarray, get_copy: bool = True) -> np.ndarray:
        """ Método para realizar convoluciones sincrónicas

        Este método realiza en forma secuencial los llamados a async_convolve y
        sync_get_results.
        El resultado es una función sincrónica que ejecuta la convolución y detiene el
        flujo del programa hasta obtener los resultados.

        Parameters
        ----------
        positions : np.ndarray
            Lista de posiciones con formato [N, n, 2/3] donde N representa la cantidad de
            "individuos", n la cantidad de fuentes virtuales y 2/3 la dimensión permitida.
        get_copy : bool
            Indica si el resultado devuelto es una copia de la memoria utilizada por la
            interfaz. Las copias son más seguras, pero también un poco más lentas.
            Para operaciones de sólo lectura en forma segura la copia no es necesaria
            (por ejemplo para calcular una métrica sobre la convolución devuelta sin
            afectar sus valores).

        Returns
        -------
        Las imágenes resultantes de la convolución o una copia de ellas
        (según el parámetro get_copy).

        """
        self.async_convolve(positions)
        return self.sync_get_results(get_copy=get_copy)

    def _background_shutdown(self):  # pragma: no cover
        raise NotImplementedError

    def _prepare(self):  # pragma: no cover
        raise NotImplementedError

    def _start_convolution(self):  # pragma: no cover
        raise NotImplementedError


class ConvolutionManagerGPU(ConvolutionManager):
    """Clase interfaz Python-CUDA para la convolución en GPU.

    Clase encargada de generar la interfaz entre Python y la librería de Catmu
    para convoluciones en GPU.

    Una vez configurada mantiene abierta una sesión de comunicación con la GPU.
    Esto permite ahorrar la creación del contexto y acelera su uso.
    Cada vez que se carga una nueva PSF la sesión anterior queda cerrada.

    El método sync_convolve realiza la convolución en forma sincrónica (pide la
    ejecución y espera los resultados). El pedido, asíncrono, se hace mediante
    el método async_convolve, mientras que la espera, sincrónica, la realiza el
    método sync_get_results.

    El uso del método asincrónico permite realizar otras tareas mientras la
    convolución se realiza en segundo plano. Por ejemplo, habilita el uso de una
    segunda GPU.

    Parameters
    ----------
    device : int
        Identificador del dispositivo utilizado (por defecto 0).
    block_size : int
        Tamaño del bloque de hilos (block_size x block_size).
        La documentación de CUDA recomienda múltiplos de 8. Por defecto 8.
    n_streams : int
        Cantidad de Streams de CUDA creados.
        Por defecto 10.
    debug : bool
        Indica si se utiliza la librería compilada con mensajes de debug.
        Por defecto False.

    Attributes
    ----------
    positions
    last_elapsed_time
    loop_counter

    """

    def __init__(self,
                 device: int = 0,
                 block_size: int = 8,
                 n_streams: int = 100,
                 debug: bool = False):

        super().__init__(debug=debug)

        # Eventos señalizados por el hilo principal
        self._main_ready = threading.Event()
        self._main_stop = threading.Event()

        # Eventos señalizados por el hilo daemon
        self._daemon_running = threading.Event()
        self._daemon_ready = threading.Event()
        self._daemon_error = threading.Event()

        # Codigo de retorno de cuda
        self._error_code = 0

        # Tiempo transcurrido entre checkpoints
        self._last_elapsed_time = 0.0
        # Cantidad de veces que se alcanzó el checkpoint
        self._loop_counter = 0

        # Configuraciones de dispositivo y división interna del trabajo
        self._device = device
        self._block_size = block_size
        self._n_streams = n_streams

    @property
    def loop_counter(self) -> int:
        return self._loop_counter

    @property
    def last_elapsed_time(self) -> float:
        return self._last_elapsed_time

    def _background_shutdown(self):
        # Detiene hilos secundarios de corridas anteriores (si existen)
        if self._daemon_running.is_set():
            logger.debug('Apagando la sesión anterior')
            self._main_stop.set()
            self._main_ready.set()
            if self._daemon_ready.wait(timeout=self._timeout) is False:
                raise TimeoutError  # pragma: no cover
            logger.debug('Sesión anterior detenida')

        if self._daemon_thread is not None:
            if self._daemon_thread.join():
                raise RuntimeError  # pragma: no cover

        # Limpia las señalizaciones existentes
        logger.debug('Flags desactivados')
        self._main_ready.clear()
        self._daemon_ready.clear()
        self._main_stop.clear()

    def _prepare(self):

        self._s_config = DevConfig(device=self._device,
                                   block_size=self._block_size,
                                   n_streams=self._n_streams)

        # Crea el hilo secundario y lo ejecuta
        logger.debug('Creando sesión en GPU')
        self._daemon_thread = threading.Thread(target=self._background_run, daemon=True)
        self._daemon_thread.start()

        # Se queda a la espera de que el hilo secundario alcance el checkpoint
        logger.debug('Esperando a que el hilo secundario esté listo')
        if self._daemon_ready.wait(timeout=self._timeout) is False:  # pragma: no cover
            raise TimeoutError

        # Cuando el hilo secundario está listo limpia su señalización
        self._daemon_ready.clear()

        # Si hubo algún error lo comunica
        if self._daemon_error.is_set():  # pragma: no cover
            logger.debug('Error de CUDA')
            raise CatmuError(code=self._error_code)
        else:
            logger.debug('Sesión creada')

    def _background_run(self):
        """ Método interno que gestiona las tareas del hilo secundario """

        try:
            # Señaliza que el hilo secundario está corriendo
            logger.debug('Hilo secundario corriendo')
            self._daemon_running.set()

            # Función de checkpoint
            def checkpoint(elapsed_time, loop_counter):
                """ Punto de control del host de CUDA antes de comenzar la convolución

                Esta función es llamada cada vez que el host de CUDA está listo para
                llevar a cabo una nueva convolución.

                La función corre dentro de este hilo secundario para evitar que congele la
                ejecución principal.

                Se utiliza la señal self._daemon_ready para indicar que el checkpoint fue
                alcanzado y se espera la señal self._main_ready para continuar.

                La función recibe del host de CUDA un indicador de tiempo transcurrido
                desde la última llamada y un contador de ciclos. Almacena dichos
                resultados en self._last_elapsed_time y self._loop_counter
                respectivamente.

                Devuelve al host de CUDA True para continuar o False para detener el
                ciclo.

                """
                logger.debug('Checkpoint alcanzado')
                self._last_elapsed_time = float(elapsed_time)
                self._loop_counter = int(loop_counter)
                logger.debug(f'Contador: {int(loop_counter)} '
                             f'Tiempo: {float(elapsed_time)}')

                self._daemon_ready.set()
                if self._main_ready.wait(timeout=self._timeout) is False:
                    raise TimeoutError  # pragma: no cover
                self._main_ready.clear()

                if self._main_stop.is_set():
                    logger.debug('Señal de STOP recibida, checkpoint devuelve False')
                    return False

                n = len(self._positions)

                self._s_image.set_data(self._results,
                                       pixel_size=self._image_pixel_size)

                self._s_positions.set_data(self._positions)

                logger.debug('El checkpoint devuelve True y la GPU continua')
                return True

            if isinstance(self._s_psf, LutPSF):
                logger.debug('Llamada a lutConvolution')
                r = self._lib.lutConvolution(callbackType(checkpoint),
                                             self._s_image,
                                             self._s_positions,
                                             self._s_psf,
                                             self._s_config)
            elif isinstance(self._s_psf, ExpressionPSF):
                logger.debug('Llamada a expressionConvolution2D')
                r = self._lib.exprConvolution2D(callbackType(checkpoint),
                                                self._s_image,
                                                self._s_positions,
                                                self._s_psf,
                                                self._s_config)
            else:
                raise TypeError  # pragma: no cover

            if r != 0:  # pragma: no cover
                self._error_code = r
                self._daemon_error.set()
            logger.debug(f'lutConvolution devolvió {r}')

        except Exception:   # pragma: no cover
            raise

        finally:
            self._daemon_ready.set()
            self._daemon_running.clear()
            logger.debug('Hilo secundario detenido')

    def _start_convolution(self):
        self._daemon_ready.clear()
        self._main_stop.clear()
        self._daemon_error.clear()
        self._main_ready.set()

    @property
    def active(self):
        return self._daemon_thread.is_alive()


class ConvolutionManagerCPU(ConvolutionManager):
    def __init__(self, open_mp: bool = True, debug: bool = False):

        super().__init__(debug=debug)

        # Usar el módulo de Multiprocessing en CPU
        self._open_mp = open_mp

    def _background_shutdown(self):
        # Detiene hilos secundarios de corridas anteriores (si existen)
        if self._daemon_thread is not None:   # pragma: no cover
            if self._daemon_thread.is_alive() is True:
                self._daemon_thread.join()
        self._daemon_ready.clear()

    def _prepare(self):
        pass

    def _start_convolution(self):
        self._daemon_ready.clear()
        if isinstance(self._s_psf, LutPSF):
            logger.debug('Llamada a cpu_lut_convolve')
            if self._open_mp is True:
                f = self._lib.cpu_lut_convolve2D_openmp
            else:
                f = self._lib.cpu_lut_convolve2D
        elif isinstance(self._s_psf, ExpressionPSF):
            logger.debug('Llamada a cpu_expr_convolve2D')
            if self._open_mp is True:
                f = self._lib.cpu_expr_convolve2D_openmp
            else:
                f = self._lib.cpu_expr_convolve2D
        else:
            raise TypeError    # pragma: no cover

        self._s_image.set_data(self._results,
                               self._image_pixel_size)

        self._s_positions.set_data(self._positions)

        self._daemon_thread = threading.Thread(target=self._background_run,
                                               args=(f,), daemon=True)
        self._daemon_thread.start()

    def _background_run(self, function):
        r = function(self._s_image,
                     self._s_positions,
                     self._s_psf)
        if r != 0:   # pragma: no cover
            raise CatmuError(f'La ejecución en CPU devolvió un código de error {r}')
        else:
            self._daemon_ready.set()
