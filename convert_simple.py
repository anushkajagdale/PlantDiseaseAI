import tensorflow as tf
import os

print('Loading model...')
try:
    model = tf.keras.models.load_model('model/plant_disease_model.keras', compile=False)
    print('Model loaded, saving as HDF5...')
    model.save('model/plant_disease_model.h5', save_format='h5')
    print('✅ Saved successfully!')
except Exception as e:
    print(f'❌ Error: {e}')