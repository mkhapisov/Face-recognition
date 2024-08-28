def draw_boxes(filename: str):
    import matplotlib.pyplot as plt
    from mtcnn.mtcnn import MTCNN
    from matplotlib.patches import Rectangle, Circle

    pixels = plt.imread(filename)
    plt.imshow(pixels)
    ax = plt.gca()

    detector = MTCNN()
    faces = detector.detect_faces(pixels)

    for face in faces:
        x0, y0, width, height = face['box']
        rect = Rectangle((x0, y0), width, height, fill=False, color='red')
        ax.add_patch(rect)

        for value in face['keypoints'].values():
            dot = Circle(value, radius=2, color='navy')
            ax.add_patch(dot)
    
    plt.show()

def get_faces(filename: str):
    import matplotlib.pyplot as plt
    from mtcnn.mtcnn import MTCNN

    pixels = plt.imread(filename)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    faces_list = []
    
    for face in faces:
        x0, y0, width, height = face['box']
        x1 = x0 + width
        y1 = y0 + height
        faces_list.append(pixels[y0:y1, x0:x1])
    
    return faces_list

def draw_faces(faces: list):
    import matplotlib.pyplot as plt

    for i in range(len(faces)):
        plt.subplot(1, len(faces), i + 1)
        plt.axis('off')
        plt.imshow(faces[i])
    
    plt.show()

if __name__ == '__main__':
    faces = get_faces('../test1.jpg')
    print(faces)
    draw_faces(faces)