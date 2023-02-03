import experiment_manager as em
import h5py
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

path_to_saved_model_file = '/path/to/the/project/saved_models_folder/model_file.h5'
'''To load the model one can specify the saved model path
or load an experiment from DB by specifying experiment_id:

exp = em.Experiment(experiment_id = 1)
path_to_saved_model_file = exp.saved_model_path + exp.output_filename + '.h5'

give 'path_to_saved_model_file' to the tf model loader

'''

# load the model
trained_model = load_model(path_to_saved_model_file)

# create an extractor model
extractor_model = tf.keras.Model(inputs=trained_model.input,
                                 outputs=[trained_model.layers[-2].output])
extractor_model.summary()

# iterate over a patch collection to get and store fingerprints
def get_patches_from_WSI(path_to_svs):
    '''This should be your usual method
    to get patches from the whole slide image.
    I preffer to use openslide library (https://openslide.org/download/):
    
    import openslide
    
    svs = openslide.OpenSlide(path_to_svs)
    patch = svs.read_region(location=[x,y], size=[width,height], level=0)
    
    '''
    
    pass


patch_collection = get_patches_from_WSI(path_to_svs)

fingerprints_array = []
with h5py.File('/path/to/the/project/fingerprints_file.h5','w') as f:
    for patch in patch_collection:
        fingerprints_vector = np.squeeze(extractor_model.predict(patch))
        fingerprints_array.append(fingerprints_vector)
        
    fingerprints_array = np.array(fingerprints_array)
    f.create_dataset('fingerprints', data = fingerprints_array)
