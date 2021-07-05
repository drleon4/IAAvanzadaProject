#ruta base al directorio de YOLO
MODEL_PATH = "yolo-coco"
#inicializar la probabilidad mínima para filtrar las detecciones débiles junto con el control al aplicar supresión no máxima 
MIN_CONF = 0.3
NMS_THRESH = 0.3
#booleano que indica si se debe utilizar NVIDIA CUDA GPU
USE_GPU = False
#definir la distancia de seguridad mínima (en píxeles) que pueden estar dos personas entre sí
MIN_DISTANCE = 50