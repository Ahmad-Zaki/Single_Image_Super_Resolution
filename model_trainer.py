from data_loader import Div2kLoader
from models import edsr, srgan_discriminator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class SrganTrainer():
    """
    SRGAN trainer class.
    
    Attributes
    ----------
    generator: Model
      The model used as a generator in SRGAN.

    desicriminator: Model
      The model used as a disrciminator in SRGAN.

    data_path: str
     The location of the training data. The gives path is expected to contain a 'HR' folder that holds high resolution images
     and a 'LR' folder that holds low resolution images, the same image must have the same name in both folders so that both LR
     and HR version are paired correctly

    lrw: int
      low resolution image width.
      Default=64

    lrh: int
      low resolution image hight.
      Default=64

    load_all_data: bool
      Whether to load the whole dataset in memory. Unless you have enough RAM in your machine, this is not recommended.
      Default=False

    learning_rate: int
      specify the learning rate used during training.
      Default=1e-4

    Methods
    -------
    train_generator(epochs: int, starting_weights: str = None, batch_size: int = 32, loss: str = "mae")
      Trains the generator model on its own.

    train_gan(self, weights_path, steps, batch_size)
      Starts the training of SRGAN.
    """

    def __init__(self, generator: Model, discriminator: Model, data_path: str, lrw: int = 64, lrh: int = 64, load_all_data: bool = False, learning_rate: float = 1e-4):
        # Input shape
        self.channels = 3
        self.lr_height = lrh
        self.lr_width = lrw
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        # Output shape
        self.hr_height = self.lr_height*4
        self.hr_width = self.lr_width*4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.generator = generator
        self.gen_optimizer = Adam(learning_rate)

        self.discriminator = discriminator
        self.disc_optimizer = Adam(learning_rate)

        self.data = Div2kLoader(data_path, load_all_data=load_all_data)
        self.learning_rate = learning_rate
        
        # Build the VGG network used in calculating the content Loss
        

    def train_generator(self, epochs: int = 150, starting_weights: str = None, batch_size: int = 32, loss: str = "mae"):
        """Trains the generator model on its own. This is important before training SRGAN because the resulting weights
        are going to be used as an initialization for the generator in SRGAN.

        Parameters
        ----------
        epochs: int
          number of training epochs.
          Default=150

        starting_weights: str
          path to initialization weights. If not specified, the generator will be initialized with random weights.
          Default=None

        batch_size: int
          number of images per batch.
          Default=32

        loss: str
          Training loss function, can be 'mae' for mean absolute error, or 'mse' for mean square error.
          Default='mae'

        Returns
        -------
        weights_path: str
          Path of the resulting weights
        """

        if starting_weights:
            print(f"Initializing '{self.generator.name}' with {starting_weights}")
            self.generator.load_weights(starting_weights)
            
        optimizer = Adam(self.learning_rate, 0.9)
        self.generator.compile(loss= [loss],
                               optimizer=optimizer)
        
        self.data.batch_size = batch_size
        #Where to save model weights:
        weights_path = f"model_weights/generator_mse/{self.generator.name}_X4_MSE-{{epoch:02d}}.h5"
        checkpoint = ModelCheckpoint(weights_path,
                                     save_best_only=False)
        
        print("Training the Generator on its own:")
        self.generator.fit(self.data,
                           epochs=epochs,
                           callbacks=[checkpoint])

        # serialize model to JSON
        with open("model_json/{self.generator.name}_X4_MSE.json", "w") as json_file:
            json_file.write(self.generator.to_json())

        print(f"training '{self.generator.name}' model completed Successfully!")
        return weights_path
      
    def train_gan(self, weights_path: str, steps: int = 2e5, batch_size: int = 16):
        """Start the training of SRGAN.

        Parameters
        ----------
        weights_path: str
          path to initialization weights.

        steps: int
          the number of training steps. At each step, the model is trained on a single batch of data.
          Default=200,000


        batch_size: int
          number of images per batch.
          Default=16
        """

        # Prepare log file:
        with open('training_history/losses.csv', 'w') as f:
          f.write("step, perc_loss, disc_loss\n")
        
        # Initialize the generator:
        self.generator.load_weights(weights_path)

        # Specify the batch size:
        self.data.batch_size = batch_size

        for step in range(1, steps + 1):
          lr, hr = self.data.load_batch()
          pl, dl = self._train_step(lr, hr)

          print(f"Step #{step}:\n    Generator loss     = {pl}\n    Discriminator loss = {dl}\n")

          #Record losses in a csv log file:
          with open('training_history/losses.csv', 'a') as f:
            f.write(f"{step}, {pl}, {dl}\n")

          #Save Weights every 200 steps
          if step % 200 == 0:
            discriminator.save_weights( f"model_weights/disc/{discriminator.name}_X4_SRGAN.h5")
            generator.save_weights(f"model_weights/gen/{generator.name}_X4_SRGAN-{step}.h5")
            print("#############\nWeights Saved\n#############\n")

    
    def _train_step(self, lr, hr):
        """SRGAN training step.
        
        Takes an LR and an HR image batch as input and returns
        the computed perceptual loss and discriminator loss.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            # Forward pass
            sr = self.generator(lr, training=True)
            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            # Compute losses
            con_loss = None
            gen_loss = None
            perc_loss = None
            disc_loss = None



        # Compute gradient of perceptual loss w.r.t. generator weights 
        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        # Compute gradient of discriminator loss w.r.t. discriminator weights 
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Update weights of generator and discriminator
        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return perc_loss, disc_loss        

    
   
    #loss_functions
   
              
if __name__ == '__main__':
    data_path = r"datasets/preprocessed_data/"
    generator = edsr()
    discriminator = srgan_discriminator()
    gan = SrganTrainer(generator,
                       discriminator,
                       data_path=data_path,
                       load_all_data=False)

    weights_path = gan.trainGenerator(epochs=150,
                                      batch_size=32)

    gan.train_gan(weights_path,
                  steps=2e5,
                  batch_size=16)