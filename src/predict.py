import matplotlib.pyplot as plt
from helpers import get_random_data, load_model

(img, img_array, img_class) = get_random_data()

print(f"Cell is originally from {img_class} class")
plt.imshow(img)

model = load_model()

prediction = model.predict(img_array, verbose=0)

print(f"Model predicted that cell is from {'uninfected' if prediction[0][0] > 0.5 else 'parasitized'} class")
print("Image from cell has been oppened in your machine. Close it to exit the program")

plt.show()
