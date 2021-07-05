# DETECCIÓN DEL DISTANCIAMIENTO SOCIAL
Proyecto sobre ML aplicado al problema del COVID-19, haciendo usos de las librerías de YOLO-COCO

**Pasos para hacer funcionar el proyecto:**

**1**. Descargar este proyecto a su directorio.

**2**. Desde su consola diríjase hasta la carpeta principal del proyecto.

**3**. Instalar librerías y dependencias (en caso de que no se instalen es necesario instalar las dependencias manualmente)
-> 

     pip install -r requirements.txt

   **Instalar ->**
   
    pip install scipy

**4**. Descargar **yolov3.weights** para el conjunto de datos COCO: https://pjreddie.com/media/files/yolov3.weights 

**5**. Poner el conjunto de datos yolov3.weights dentro de la carpeta **yolo-coco**

**6. Para la ejecución:**

  **6.1**. Para ejecutar un video: ->
  
      python social_distance_detector.py --input prueba1.mp4
  
  **6.2**. Para ejecutar un video y exportar el resultado en formato **.AVI** -> 
  
      python social_distance_detector.py --input prueba2.mp4 --output salidaPrueba2.avi
  

**NOTA:** Se adjunta dos videos (prueba1.mp4 y prueba2.mp4) dentro del proyecto con los cuales se realizó las pruebas, pero se puede ejecutar con videos a nuestra conveniencia.


**EL PROCESO DE EXPORTACIÓN DEL RESULTADO (a .AVI) ES TARDÍO POR LOS FPS QUE EL MODELO YOLO-COCO TIENE QUE PROCESAR**
