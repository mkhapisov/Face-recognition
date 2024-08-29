import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle, Circle
from numpy import ndarray


def draw_boxes(filename: str, path: str = None):
    # it's an additional function for drawing detection boxes and some key points
    # this function isn't used in code, it's needed only for debugging
    pixels = plt.imread(filename)
    plt.imshow(pixels)
    ax = plt.gca()

    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    for face in faces:
        x0, y0, width, height = face["box"]
        rect = Rectangle((x0, y0), width, height, fill=False, color="red")
        ax.add_patch(rect)

        for value in face["keypoints"].values():
            dot = Circle(value, radius=2, color="navy")
            ax.add_patch(dot)

    if path is None:
        plt.show()
    else:
        plt.savefig(path + f"{filename}_boxed.jpg", dpi=500)
        plt.close()


def get_faces(filename: str) -> list[ndarray]:
    pixels = plt.imread(filename)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    faces_list = []

    for face in faces:
        x0, y0, width, height = face["box"]
        x1 = x0 + width
        y1 = y0 + height
        faces_list.append(pixels[y0:y1, x0:x1])

    return faces_list


def draw_faces(faces: list[ndarray], path: str = None):
    # it's an additional function for showing faces getting from get_faces
    # this function isn't used in code, it's needed only for debugging
    for i in range(len(faces)):
        plt.subplot(1, len(faces), i + 1)
        plt.axis("off")
        plt.imshow(faces[i])

    if path is None:
        plt.show()
    else:
        plt.savefig(path + "faces.jpg", dpi=500)
        plt.close()
