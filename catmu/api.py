import logging
import pathlib
import threading
import ctypes
from typing import Tuple
import numpy as np

from catmu.structures import sImage2d, sPositions2d, sPSF, sConfig

# Configuración del módulo Logging
logging.basicConfig(format="%(threadName)s : %(message)s", level='INFO')
logger = logging.getLogger(__name__)


# Punteros a cada tipo de estructura
pImage2d = ctypes.POINTER(sImage2d)
pPositions2d = ctypes.POINTER(sPositions2d)
pPSF = ctypes.POINTER(sPSF)
pConfig = ctypes.POINTER(sConfig)
callback_type = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_double, ctypes.c_ulong)


class ConvolutionManagerInternalError(Exception):
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
    """Clase interfaz Python-CUDA para la convolución LUT.

    Clase encargada de generar la interfaz entre Python y la librería de
    convolución por LUT.

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
    sub_pixel : int
        Cantidad de píxeles trabajado por cada Thread. Por defecto 1.
        Cada Thread calcula sub_pixel**2 píxeles.
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
                 patch_length: int = 1,
                 debug: bool = False):

        # Selecciona la biblioteca a utilizar (modo debug o no)
        if debug is True:
            self._lib_name = pathlib.Path(pathlib.Path(__file__).parent / f'cuda/libConvolveLUTd.so')
        else:
            self._lib_name = pathlib.Path(pathlib.Path(__file__).parent / f'cuda/libConvolveLUT.so')

        # Si la biblioteca no existe intenta compilarla
        if self._lib_name.exists() is False:
            print(f"Compilando {self._lib_name.name}...")
            import subprocess
            make_command = f'make -C {pathlib.Path(__file__).parent / "cuda"} all'
            subprocess.run(make_command, shell=True)
            if self._lib_name.exists() is False:
                raise FileNotFoundError(f'La biblioteca {self._lib_name} no pudo ser compilada.')
            print("Compilación exitosa")

        # Carga la biblioteca
        self._lib = ctypes.CDLL(self._lib_name)

        # Configura los tipos recibidos y el tipo devuelto
        self._lib.lutConvolution2D.argtypes = [callback_type, pImage2d, pPositions2d, pPSF, pConfig]
        self._lib.lutConvolution2D.restype = ctypes.c_int

        self._lib.cpu_convolve.argtypes = [pImage2d, pPositions2d, pPSF]
        self._lib.cpu_convolve.restype = ctypes.c_int

        # Arrays de Numpy con la memoria ocupada en la CPU por la PSF, posiciones y resultados
        self._psf = None
        self._positions = None
        self._results = None

        # Configuraciones de tamaños de píxel
        self._image_size = None
        self._psf_pixel_size = None
        self._image_pixel_size = None

        # Eventos señalizados por el hilo principal
        self._main_ready = threading.Event()
        self._main_stop = threading.Event()

        # Eventos señalizados por el hilo daemon
        self._daemon_running = threading.Event()
        self._daemon_ready = threading.Event()
        self._daemon_cuda_error = threading.Event()

        # Codigo de retorno de cuda
        self._cuda_error_code = 0

        # Tiempo transcurrido entre checkpoints
        self._last_elapsed_time = 0.0
        # Cantidad de veces que se alcanzó el checkpoint
        self._loop_counter = 0

        # Configuraciones de dispositivo y división interna del trabajo
        self._device = device
        self._sub_pixel = patch_length
        self._block_size = block_size
        self._n_streams = n_streams

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def last_elapsed_time(self):
        return self._last_elapsed_time

    @property
    def loop_counter(self):
        return self._loop_counter

    def prepare(self,
                psf: np.ndarray,
                image_size: Tuple[int, int] = (64, 64),
                image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                psf_pixel_size: Tuple[float, float] = (1.0, 1.0)):
        """ Metodo utilizado para configurar una sesión de trabajo con una PSF dada.

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

        # Configura la PSF
        if isinstance(psf, (tuple, list, np.ndarray)):
            self._psf = np.asarray(psf, dtype=ctypes.c_float, order='c')
        else:
            raise TypeError

        # Configura el tamaño de la imagen resultante y los tamaños de píxeles
        self._image_size = image_size
        self._image_pixel_size = image_pixel_size
        self._psf_pixel_size = psf_pixel_size

        # Detiene hilos secundarios de corridas anteriores (si existen)
        if self._daemon_running.is_set():
            logger.debug('Apagando la sesión anterior')
            self._main_stop.set()
            self._main_ready.set()
            self._daemon_ready.wait()
            logger.debug('Sesión anterior detenida')

        # Limpia las señalizaciones existentes
        logger.debug('Flags desactivados')
        self._main_ready.clear()
        self._daemon_ready.clear()
        self._main_stop.clear()

        # Crea el hilo secundario y lo ejecuta
        logger.debug('Creando sesión en GPU')
        x = threading.Thread(target=self._session_run, args=(psf,), daemon=True)
        x.start()

        # Se queda a la espera de que el hilo secundario alcance el checkpoint
        logger.debug('Esperando a que el hilo secundario esté listo')
        self._daemon_ready.wait()

        # Cuando el hilo secundario está listo limpia su señalización
        self._daemon_ready.clear()

        # Si hubo algún error lo comunica
        if self._daemon_cuda_error.is_set():
            logger.debug('Error de CUDA')
            raise ConvolutionManagerInternalError(code=self._cuda_error_code)
        else:
            logger.debug('Sesión creada')

    def async_convolve(self, positions: np.ndarray):
        """ Método que comienza el proceso de convolución en forma asincrónica

        Este método da la señal al hilo secundario para que comience la tarea de convolucionar
        la lista de posiciones indicadas.

        La función devuelve el control inmediatamente y deja a cargo del usuario la espera de
        los resultados.

        Se habilita de esta forma la posibilidad de ejecutar tareas en dos GPU simultaneamente.
        Bastará con tener dos instancias de ConvolutionManager apuntando a dispositivos diferentes
        y ejecutar en ambas convoluciones asincrónicas.

        El método sync_get_results marca la barrera, requiriendo que la convolución finalice para
        continuar con la ejecución del hilo principal.

        Para convoluciones sincronicas ver: sync_convolve

        Parameters
        ----------
        positions : np.ndarray
            Lista de posiciones con formato [N, n, 2] donde N representa la cantidad de "individuos",
            n la cantidad de fuentes virtuales y 2 la dimensión permitida (próximamente 3).

        """
        logger.debug('Iniciando convolución')
        if isinstance(positions, (tuple, list, np.ndarray)):
            self._positions = np.asarray(positions, dtype=ctypes.c_float, order='c')
        else:
            raise TypeError

        logger.debug('Esperando a que el hilo secundario llegue al checkpoint')
        self._daemon_ready.clear()
        self._main_stop.clear()
        self._daemon_cuda_error.clear()
        self._main_ready.set()

    def sync_get_results(self, get_copy: bool = True):
        """ Método para esperar resultados generados por async_convolve

        Este método sirve como barrera para detener la ejecución del hilo principal hasta que la
        convolución haya finalizado y además devuelve los resultados obtenidos.

        Para convoluciones sincrónicas ver: sync_convolve

        Parameters
        ----------
        get_copy : bool
            Indica si el resultado devuelto es una copia de la memoria utilizada por la interfaz.
            Las copias son más seguras, pero también un poco más lentas.
            Para operaciones de sólo lectura en forma segura la copia no es necesaria (por ejemplo
            para calcular una métrica sobre la convolución devuelta sin afectar sus valores).

        Returns
        -------
        Las imágenes resultantes de la convolución o una copia de ellas (según el parámetro get_copy).

        """
        self._daemon_ready.wait()
        self._daemon_ready.clear()

        if self._daemon_cuda_error.is_set():
            logger.debug('Error de CUDA')
            raise ConvolutionManagerInternalError(code=self._cuda_error_code)
        else:
            logger.debug('Convolución finalizada')
            if get_copy is True:
                return np.copy(self._results)
            else:
                return self._results

    def sync_convolve(self, positions: np.ndarray, get_copy: bool = True):
        """ Método para realizar convoluciones sincrónicas

        Este método realiza en forma secuencial los llamados a async_convolve y sync_get_results.
        El resultado es una función sincrónica que ejecuta la convolución y detiene el flujo del
        programa hasta obtener los resultados.

        Parameters
        ----------
        positions : np.ndarray
            Lista de posiciones con formato [N, n, 2] donde N representa la cantidad de "individuos",
            n la cantidad de fuentes virtuales y 2 la dimensión permitida (próximamente 3).
        get_copy : bool
            Indica si el resultado devuelto es una copia de la memoria utilizada por la interfaz.
            Las copias son más seguras, pero también un poco más lentas.
            Para operaciones de sólo lectura en forma segura la copia no es necesaria (por ejemplo
            para calcular una métrica sobre la convolución devuelta sin afectar sus valores).

        Returns
        -------
        Las imágenes resultantes de la convolución o una copia de ellas (según el parámetro get_copy).


        """
        self.async_convolve(positions)
        return self.sync_get_results(get_copy=get_copy)

    def cpu_convolve(self, psf: np.ndarray, positions: np.ndarray,
                     image_size: Tuple[int, int] = (64, 64),
                     image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                     psf_pixel_size: Tuple[float, float] = (1.0, 1.0),
                     get_copy: bool = True):
        """ Método para realizar la convolución primitivamente en CPU

        Esté método utiliza la función cpu_convolve de la biblioteca lutConvolution.so para realizar
        una convolución primitiva (tiene 4 bucles for anidados). Dicha función utiliza a su vez un
        simulador de interpolación de texturas basado en la documentación de CUDA.

        Parameters
        ----------

        psf : np.ndarray
            PSF a cargar en memoria.
        positions : np.ndarray
            Lista de posiciones con formato [N, n, 2] donde N representa la cantidad de "individuos",
            n la cantidad de fuentes virtuales y 2 la dimensión permitida (próximamente 3).
        image_size : Tuple[int, int]
            Tamaño de la imagen resultante
        image_pixel_size : Tuple[float, float]
            Tamaño del pixel de la imagen
        psf_pixel_size : Tuple[float, float]
            Tamaño del pixel de la psf
        get_copy : bool
            Indica si el resultado devuelto es una copia de la memoria utilizada por la interfaz.
            Las copias son más seguras, pero también un poco más lentas.
            Para operaciones de sólo lectura en forma segura la copia no es necesaria (por ejemplo
            para calcular una métrica sobre la convolución devuelta sin afectar sus valores).

        Returns
        -------
        Las imágenes resultantes de la convolución o una copia de ellas (según el parámetro get_copy).

        """
        # Configura la lista de posiciones
        if isinstance(positions, (tuple, list, np.ndarray)):
            self._positions = np.asarray(positions, dtype=ctypes.c_float, order='c')
        else:
            raise TypeError

        # Configura la PSF
        if isinstance(psf, (tuple, list, np.ndarray)):
            self._psf = np.asarray(psf, dtype=ctypes.c_float, order='c')
        else:
            raise TypeError

        # Configura el tamaño de la imagen resultante y los tamaños de píxeles
        self._image_size = image_size
        self._image_pixel_size = image_pixel_size
        self._psf_pixel_size = psf_pixel_size

        # Genera las estructuras necesarias para la interfaz con la biblioteca
        s_psf = sPSF.create(psf_data=self._psf,
                            pixel_width=self._psf_pixel_size[1],
                            pixel_height=self._psf_pixel_size[0])

        s_image = sImage2d()
        s_positions = sPositions2d()

        n = len(self._positions)
        self._results = np.zeros((n, image_size[0], image_size[1]), dtype=ctypes.c_float, order='c')

        s_image.set_data(self._results,
                         pixel_width=self._image_pixel_size[1],
                         pixel_height=self._image_pixel_size[0])

        s_positions.set_data(self._positions)

        # Ejecuta la función cpu_convolve de la biblioteca
        r = self._lib.cpu_convolve(s_image, s_positions, s_psf)

        # Captura un posible error o devuelve el resultado obtenido
        if r != 0:
            raise ConvolutionManagerInternalError("Error en el método cpu_convolve de lutConvolution.so")
        else:
            if get_copy is True:
                return np.copy(self._results)
            else:
                return self._results

    def _session_run(self, psf):
        """ Método interno que gestiona las tareas del hilo secundario """

        try:
            # Señaliza que el hilo secundario está corriendo
            logger.debug('Hilo secundario corriendo')
            self._daemon_running.set()

            # Genera las estructuras necesarias para la interfaz con la biblioteca
            self._psf = np.asarray(psf, dtype=ctypes.c_float, order='c')

            s_psf = sPSF.create(psf_data=self._psf,
                                pixel_width=self._psf_pixel_size[1],
                                pixel_height=self._psf_pixel_size[0])

            s_config = sConfig.create(device=self._device,
                                      sub_pixel=self._sub_pixel,
                                      block_size=self._block_size,
                                      n_streams=self._n_streams)

            s_image = sImage2d()
            s_positions = sPositions2d()

            # Función de checkpoint
            def checkpoint(elapsed_time, loop_counter):
                """ Función interna llamada desde el host de CUDA antes de comenzar la convolución

                Esta función es llamada cada vez que el host de CUDA está listo para llevar a cabo
                una nueva convolución.

                La función corre dentro de este hilo secundario para evitar que congele la ejecución
                principal.

                Se utiliza la señal self._daemon_ready para indicar que el checkpoint fue alcanzado
                y se espera la señal self._main_ready para continuar.

                La función recibe del host de CUDA un indicador de tiempo transcurrido desde la última
                llamada y un contador de ciclos. Almacena dichos resultados en self._last_elapsed_time
                y self._loop_counter respectivamente.

                Devuelve al host de CUDA True para continuar o False para detener el ciclo.

                """
                logger.debug('Checkpoint alcanzado')
                self._last_elapsed_time = float(elapsed_time)
                self._loop_counter = int(loop_counter)
                logger.debug(f'Contador: {int(loop_counter)} Tiempo: {float(elapsed_time)}')

                self._daemon_ready.set()
                self._main_ready.wait()
                self._main_ready.clear()

                if self._main_stop.is_set():
                    logger.debug('Señal de STOP recibida, checkpoint devuelve False')
                    return False

                n = len(self._positions)
                self._results = np.zeros((n, self._image_size[0], self._image_size[1]),
                                         dtype=ctypes.c_float, order='c')

                s_image.set_data(self._results,
                                 pixel_width=self._image_pixel_size[1],
                                 pixel_height=self._image_pixel_size[0])

                s_positions.set_data(self._positions)

                logger.debug('El checkpoint devuelve True y la GPU continua')
                return True

            logger.debug('Llamada a luConvolution2D')
            r = self._lib.lutConvolution2D(callback_type(checkpoint), s_image, s_positions, s_psf, s_config)
            if r != 0:
                self._cuda_error_code = r
                self._daemon_cuda_error.set()
            logger.debug(f'lutConvolution2D devolvió {r}')

        finally:
            self._daemon_ready.set()
            self._daemon_running.clear()
            logger.debug('Hilo secundario detenido')
