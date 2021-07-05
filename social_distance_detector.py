# importar los paquetes necesarios
from pinoydatascientist import social_distancing_config as config
from pinoydatascientist.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# Construir los argumentos parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Cargar las etiquetas de la clase COCO con la que se entrenó nuestro modelo YOLO
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derivar las rutas a los pesos YOLO y la configuración del modelo
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# cargamos su detección de objetos YOLO entrenada en el conjunto de datos COCO (80 clases)
print("[INFO] cargando YOLO desde memoria...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# comprobar si vamos a usar GPU
if config.USE_GPU:
	# establecer CUDA como el backend y el destino preferibles
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determinar solo los nombres de las capas de * salida * que necesitamos de YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# inicializar la secuencia de video y el puntero para generar un archivo de video
print("[INFO] accediendo a video...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# recorrer los fotogramas de la secuencia de vídeo
while True:
	# leer el siguiente fotograma del archivo
	(grabbed, frame) = vs.read()

	# si no se agarró el marco, entonces habremos llegado al final
	# de la corriente
	if not grabbed:
		break

	# cambiar el tamaño del marco y luego detectar personas (y solo personas) en él
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# inicializar el conjunto de índices que violan el mínimo social
	# distancia
	violate = set()

	# asegúrese de que haya * al menos * dos detecciones de personas (requerido en
	# para calcular nuestros mapas de distancia por pares)
	if len(results) >= 2:
		# extraer todos los centroides de los resultados y calcular el
		# Distancias euclidianas entre todos los pares de centroides
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# bucle sobre el triangular superior de la matriz de distancias
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# compruebe si la distancia entre dos
				# pares de centroides es menor que el número configurado
				# de píxeles
				if D[i, j] < config.MIN_DISTANCE:
					# actualizar nuestro conjunto de infracciones con los índices de
					# los pares de centroides
					violate.add(i)
					violate.add(j)

	# recorrer los resultados
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extraer el cuadro delimitador y las coordenadas del centroide, luego
		# inicializar el color de la anotación
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# si el par de índices existe dentro del conjunto de violaciones, entonces
		# actualizar el color
		if i in violate:
			color = (0, 0, 255)

		# dibujar (1) un cuadro delimitador alrededor de la persona y (2) el
		# coordenadas centroides de la persona,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# Dibujar el número total de violaciones de distanciamiento social en el
	# marco de salida
	text = "Violacion de Distanciamiento: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# compruebe si el marco de salida debe mostrarse en nuestra
	# pantalla
	if args["display"] > 0:
		
		# mostrar el marco de salida

		cv2.imshow("Distanciamiento SOcial", frame)
		key = cv2.waitKey(1) & 0xFF

		# si se presionó la tecla `q`, salga del bucle
		if key == ord("q"):
			break

	# si se ha proporcionado una ruta de archivo de video de salida y el video
	# el escritor no se ha inicializado, hágalo ahora
	if args["output"] != "" and writer is None:
		# Inicializar nuestro video escritor
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# si el escritor de video no es Ninguno, escriba el cuadro en la salida
	# video file
	if writer is not None:
		writer.write(frame)

# comandos para ejecutar el dectector
# python social_distance_detector.py --input prueba1.mp4

# PARA EJECUTAR Y EXPORTAR EL RESULTADO EN FORMATO AVI
# python social_distance_detector.py --input prueba2.mp4 --output salidaPrueba2.avi