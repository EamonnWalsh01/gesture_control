import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
# Baixando a imagem a ser analisada.


# Carregando a imagem usando OpenCV.
img = cv2.imread("organized_dataset/test/07_ok/frame_00_07_0006.png")

# Obtendo a largura e a altura da imagem.
img_width = img.shape[1]
img_height = img.shape[0]

# Criando uma figura e um conjunto de eixos.
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
ax.imshow(img[...,::-1])
plt.show()
# Inicializando os módulos Pose e Drawing do MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(static_image_mode=True) as pose:
    """
    Esta função utiliza a biblioteca MediaPipe para detectar e desenhar 'landmarks'
    (pontos de referência) em uma imagem. Os 'landmarks' são pontos de interesse
    que representam diferentes partes do corpo detectadas na imagem.

    Args:
        static_image_mode: um booleano para informar se a imagem é estática (True) ou sequencial (False).
    """

    # Faz uma cópia da imagem original.
    annotated_img = img.copy()

    # Processa a imagem.
    results = pose.process(img)

    # Define o raio do círculo para desenho dos 'landmarks'.
    # O raio é escalado como uma porcentagem da altura da imagem.
    circle_radius = int(.007 * img_height)

    # Especifica o estilo de desenho dos 'landmarks'.
    point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)

    # Desenha os 'landmarks' na imagem.
    mp_drawing.draw_landmarks(annotated_img,
                              landmark_list=results.pose_landmarks,
                              landmark_drawing_spec=point_spec)

# Cria uma figura e um conjunto de eixos.
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
ax.imshow(annotated_img[:, :, ::-1])
plt.show()
# Faz uma cópia da imagem original.
annotated_img = img.copy()

# Especifica o estilo de desenho das conexões dos landmarks.
line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

# Desenha tanto os pontos dos landmarks quanto as conexões.
mp_drawing.draw_landmarks(
    annotated_img,
    landmark_list=results.pose_landmarks,
    connections=mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=point_spec,
    connection_drawing_spec=line_spec
    )

# Cria uma figura e um conjunto de eixos.
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')
ax.imshow(annotated_img[...,::-1])
plt.show()
