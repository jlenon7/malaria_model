import time
import calendar
import pandas as pd
import matplotlib.pyplot as plt
from constants import MODEL_NAME
from helpers import path, get_data, load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

(train_image_gen, test_image_gen) = get_data()

model = load_model()

model.fit(
  train_image_gen,
  epochs = 20,
  validation_data = test_image_gen,
  callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
    TensorBoard(
      log_dir=path.board(f'fit-{calendar.timegm(time.gmtime())}'),
      histogram_freq=1,
      write_graph=True,
      write_images=True,
      update_freq='epoch',
      profile_batch=2,
      embeddings_freq=1
    )
  ]
)

metrics = pd.DataFrame(model.history.history)

plt.figure(figsize=(10,8))
metrics[['loss', 'val_loss']].plot()
plt.savefig(path.plots('loss.png'))

plt.figure(figsize=(10,8))
metrics[['accuracy', 'val_accuracy']].plot()
plt.savefig(path.plots('accuracy.png'))

predictions = model.predict(test_image_gen)
predictions = predictions > 0.5

confusion_matrix_result = confusion_matrix(test_image_gen.classes, predictions)
classification_report_result = classification_report(test_image_gen.classes, predictions)

print()
print('Confusion Matrix:')
print(confusion_matrix_result)
print()
print('Classification Report:')
print(classification_report_result)

print()
print('Saving the model at', path.storage(MODEL_NAME))
model.save(path.storage(MODEL_NAME))
