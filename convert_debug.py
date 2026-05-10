import sys
print('Python version:', sys.version)
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Loading model...')
try:
    model = tf.keras.models.load_model('model/plant_disease_model.keras', compile=False)
    print('Model loaded successfully')
    print('Saving as HDF5...')
    model.save('model/plant_disease_model.h5', save_format='h5')
    print('✅ Saved successfully!')
except Exception as e:
    print('❌ Error:', str(e))
    import traceback
    traceback.print_exc()