import scipy.io as sio
import numpy as np
import pickle
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


######################################################## 

class SwitchLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer that switches between different sets of weights
    (kernels and biases) based on a session value input. This layer allows
    for dynamically changing the computation based on the session.

    Attributes:
        input_dimensions (list of int): List of input dimensions for each session.
        output_dimensions (list of int): List of output dimensions for each session.
        kernels (list of tf.Variable): List of kernel weights for each session.
        biases (list of tf.Variable): List of bias weights for each session.
        dropout_rate (float): Dropout rate for regularization.

    Methods:
        call(inputs): Applies the layer to the inputs, selecting the appropriate
                      weights based on the session value.
    """

    def __init__(self, input_dimensions, output_dimensions, dropout_rate=0, **kwargs):
        """
        Initializes the SwitchLayer with the given input and output dimensions.

        Args:
            input_dimensions (list of int): List of input dimensions for each session.
            output_dimensions (list of int): List of output dimensions for each session.
            dropout_rate (float): Dropout rate for regularization. Default is 0.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super(SwitchLayer, self).__init__(**kwargs)

        # Handle different types of input_dimensions and output_dimensions
        if isinstance(input_dimensions, int) and isinstance(output_dimensions, int):
            input_dimensions = [input_dimensions]
            output_dimensions = [output_dimensions]
        elif isinstance(input_dimensions, int) and isinstance(output_dimensions, list):
            input_dimensions = [input_dimensions for _ in output_dimensions]
        elif isinstance(input_dimensions, list) and isinstance(output_dimensions, int):
            output_dimensions = [output_dimensions for _ in input_dimensions]
        elif isinstance(input_dimensions, list) and isinstance(output_dimensions, list):
            pass
        else:
            raise ValueError('Invalid inputs for input/output dimensions')

        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.dropout_rate = dropout_rate
        
        self.kernels = []
        self.biases = []
        # Loop over input and output dimensions to initialize weights for each session
        for isession, (input_, output_) in enumerate(zip(input_dimensions, output_dimensions)):
            # Add kernel weights for each session
            self.kernels += [self.add_weight(f"kernel_%d" % isession,
                                             shape=[input_, output_],
                                             initializer='glorot_uniform')]
            # Add bias weights for each session
            self.biases += [self.add_weight(f"bias_%d" % isession, 
                                            shape=[output_],
                                            initializer='zeros')]

    # @tf.function  # Optional: Use tf.function to ensure the method runs as a graph function
    def call(self, inputs):
        """
        Applies the SwitchLayer to the inputs, selecting the appropriate
        weights (kernel and bias) based on the session value.

        Args:
            inputs (tuple): A tuple containing the session value and the neural input.
                            - session_value (tf.Tensor): A scalar tensor indicating the session.
                            - neural_input (tf.Tensor): The input tensor with shape (batch_size, timesteps, input_dim).

        Returns:
            tf.Tensor: The output tensor after applying the selected weights.
        """
        session_value, neural_input = inputs  # Unpack inputs
        # Optionally, add dropout if dropout_rate is greater than 0
        if self.dropout_rate > 0:
            # Apply dropout to neural_input for each input dimension
            dropout_fns = [tf.keras.layers.Dropout(self.dropout_rate, 
                            noise_shape=(1, 1, inp_dim))(np.ones((1, 1, inp_dim)).astype(np.float32)) for inp_dim in self.input_dimensions]
            # Use tf.switch_case to select the appropriate dropout function based on session_value
            neural_input = tf.switch_case(session_value, branch_fns={i: (lambda k=dropout_fns[i]: k) for i in range(len(dropout_fns))}) * neural_input
        
        # Select the appropriate kernel based on session_value using tf.switch_case
        kernel = tf.switch_case(session_value, branch_fns={i: (lambda k=self.kernels[i]: k) for i in range(len(self.kernels))})
        
        # Select the appropriate bias based on session_value using tf.switch_case
        bias = tf.switch_case(session_value, branch_fns={i: (lambda k=self.biases[i]: k) for i in range(len(self.biases))})
        
        # Perform matrix multiplication (einsum) and add bias
        # tf.einsum is used for flexible tensor operations, here it performs the equivalent of tf.matmul
        return tf.einsum('ntb,ba->nta', neural_input, kernel) + bias

# GRU cells
class ControllerGeneratorSmoothInputCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, controller_hidden, generator_hidden, input_dim, factors, **kwargs):
        self.controller_hidden = controller_hidden
        self.generator_hidden = generator_hidden
        self.factors = factors
        self.input_dim = input_dim

        self.controller_cell = tf.keras.layers.GRUCell(controller_hidden, name='controller_cell')
        self.input_map_mean = tf.keras.layers.Dense(input_dim)  # , activation='relu')
        self.input_map_sd = tf.keras.layers.Dense(input_dim)
        self.input_map_sd_noise_prev = tf.keras.layers.Dense(input_dim)

        self.generator_cell = tf.keras.layers.GRUCell(generator_hidden, name='generator_cell')
        # self.generator_cell = tf.keras.layers.SimpleRNNCell(generator_hidden, activation=None, name='generator_cell')
        self.factor_map = tf.keras.layers.Dense(factors)
        self.eps = 1e-6
        super(ControllerGeneratorSmoothInputCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return [self.controller_hidden, self.generator_hidden, self.factors, self.input_dim]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        controller_hidden, generator_hidden, ft, noise_prev = states

        ct, controller_hidden = self.controller_cell(tf.concat([inputs, ft], axis=-1), controller_hidden)
        it_mean = self.input_map_mean(ct)
        '''
        it_mean = self.input_map_mean(ct) + self.eps
        u = tf.random.uniform([it_mean.shape[-1]])
        exp1rand = -tf.math.log(u)
        it = it_mean * exp1rand
        print(u.shape, exp1rand.shape, it.shape)
        '''
        Rii = tf.nn.relu(self.input_map_sd(ct)) + 0.5 # the +1 is a hack to avoid very small values...
        Rij = self.input_map_sd_noise_prev(ct)  # can be negative as well ?

        # print('Softhresholding the input')
        # it_mean = tf.nn.relu(it_mean - 1) - tf.nn.relu(-it_mean - 1)

        noise_sample = tf.random.normal(tf.shape(it_mean))
        noise_new = (noise_sample - noise_prev * Rij) / Rii
        it = it_mean + noise_new


        # it = it_mean + noise_sample * it_sd + noise_prev * it_sd_prev

        gt, generator_hidden = self.generator_cell(it, generator_hidden)
        ft = self.factor_map(gt)

        states = [controller_hidden, generator_hidden, ft, noise_new]
        output = [ft, it, it_mean, Rii**2]

        return output, states



class ControllerGeneratorSmoothInputCell2(tf.keras.layers.AbstractRNNCell):

    def __init__(self, controller_hidden, generator_hidden, input_dim, factors, **kwargs):
        self.controller_hidden = controller_hidden
        self.generator_hidden = generator_hidden
        self.factors = factors
        self.input_dim = input_dim
        self.noise_smoothen = 0.99

        self.controller_cell = tf.keras.layers.GRUCell(controller_hidden, name='controller_cell')
        self.input_map_mean = tf.keras.layers.Dense(input_dim)  # , activation='relu')
        self.input_map_sd = tf.keras.layers.Dense(input_dim)
        self.input_map_sd_noise_prev = tf.keras.layers.Dense(input_dim)

        self.generator_cell = tf.keras.layers.GRUCell(generator_hidden, name='generator_cell')
        # self.generator_cell = tf.keras.layers.SimpleRNNCell(generator_hidden, activation=None, name='generator_cell')
        self.factor_map = tf.keras.layers.Dense(factors)
        self.eps = 1e-6
        super(ControllerGeneratorSmoothInputCell2, self).__init__(**kwargs)

    @property
    def state_size(self):
        return [self.controller_hidden, self.generator_hidden, self.factors, self.input_dim]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        controller_hidden, generator_hidden, ft, it_prev = states

        ct, controller_hidden = self.controller_cell(tf.concat([inputs, ft], axis=-1), controller_hidden)
        it_mean = self.input_map_mean(ct)
        it_sd = tf.exp(self.input_map_sd(ct)) # the +1 is a hack to avoid very small values...
        noise_sample = tf.random.normal(tf.shape(it_mean))

        it = it_mean + noise_sample * it_sd
        it_new = it_prev * self.noise_smoothen + (1 - self.noise_smoothen) * it

        gt, generator_hidden = self.generator_cell(it, generator_hidden)
        ft = self.factor_map(gt)

        states = [controller_hidden, generator_hidden, ft, it_new]
        output = [ft, it, it_mean, it_sd]

        return output, states

class NoisyGRUCell(tf.keras.layers.GRUCell):
    def __init__(self, units, noise_stddev=0.1, **kwargs):
        super().__init__(units, **kwargs)
        self.noise_stddev = noise_stddev

    def call(self, inputs, states, training=None):

        if training:
            states_with_noise = [s + tf.random.normal(tf.shape(s), stddev=self.noise_stddev)
                                for s in states]
        else:
            states_with_noise = [s for s in states]
        return super().call(inputs, states_with_noise, training=training)

########################################################
# Model components

def build_encoder_ic(neurons, control, n_hidden_decode, n_sessions, dropout_rate):
    '''Recognition network. '''

    print('Building initial condition encoder')

    xInput = tf.keras.Input(shape=(None, None), name='xInput')
    SessionInput = tf.keras.Input(shape=(), dtype=tf.int32, name='SessionInput')

    xInput_transformed = SwitchLayer(neurons, n_hidden_decode, dropout_rate)((SessionInput, xInput))
    # xInput_transformed = tf.keras.layers.Dropout(dropout_rate, 
    #                                                 noise_shape=(1, 1, neurons))(xInput)

    xf = tf.keras.layers.GRU(n_hidden_decode)(xInput_transformed)
    xb = tf.keras.layers.GRU(n_hidden_decode, go_backwards=True)(xInput_transformed)
    e = tf.concat([xf, xb], axis=1)
    e = tf.keras.layers.Dropout(dropout_rate)(e)

    mu_phi = tf.keras.layers.Dense(control)(e)  # posterior over kinematics z (latent variables)
    sig_phi = tf.exp(0.5 * tf.keras.layers.Dense(control, activation=None)(e))
    encoder_ic = tf.keras.Model(inputs=[xInput, SessionInput],
                                        outputs=[mu_phi, sig_phi])
    return encoder_ic


def build_encoder_bias(neurons, n_inputs, n_hidden_decode, dropout_rate):

    print('Building bias encoder')

    xInput = tf.keras.Input(shape=(None, None), name='xInput')
    SessionInput = tf.keras.Input(shape=(), dtype=tf.int32, name='SessionInput')

    xInput_transformed = SwitchLayer(neurons, n_hidden_decode, dropout_rate)((SessionInput, xInput))

    xf = tf.keras.layers.GRU(n_hidden_decode)(xInput_transformed)
    xb = tf.keras.layers.GRU(n_hidden_decode, go_backwards=True)(xInput_transformed)
    e = tf.concat([xf, xb], axis=1)
    e = tf.keras.layers.Dropout(dropout_rate)(e)
    
    mu_phi = tf.keras.layers.Dense(n_inputs, activation=None)(e) 
    sig_phi = tf.exp(0.5 * tf.keras.layers.Dense(n_inputs, activation=None)(e))
    print('mu_phi', mu_phi.shape, 'sig_phi', sig_phi.shape)

    model_encoder_bias = tf.keras.Model(inputs=[xInput, SessionInput],
                                        outputs=[mu_phi, sig_phi])
    return model_encoder_bias


def build_encoder_tv_input(neurons, n_inputs, n_hidden_decode, dropout_rate):

    print('Building time-varying input encoder')

    xInput = tf.keras.Input(shape=(None, None), name='xInput')
    SessionInput = tf.keras.Input(shape=(), dtype=tf.int32, name='SessionInput')

    xInput_transformed = SwitchLayer(neurons, n_hidden_decode, dropout_rate)((SessionInput, xInput))

    # Bidirectional GRU
    xf = tf.keras.layers.GRU(n_hidden_decode, return_sequences=True)(xInput_transformed)
    xb = tf.keras.layers.GRU(n_hidden_decode, go_backwards=True, return_sequences=True)(xInput_transformed)
    e = tf.concat([xf, xb], axis=2)
    
    # Standard GRU to identify the input
    e = tf.keras.layers.GRU(n_hidden_decode, return_sequences=True)(e)
    e = tf.keras.layers.Dropout(dropout_rate)(e)
    
    mu_phi = tf.keras.layers.Dense(n_inputs, activation=None)(e) 
    sig_phi = tf.exp(0.5 * tf.keras.layers.Dense(n_inputs, activation=None)(e))
    print('mu_phi', mu_phi.shape, 'sig_phi', sig_phi.shape)

    model_encoder_tv_input = tf.keras.Model(inputs=[xInput, SessionInput],
                                        outputs=[mu_phi, sig_phi])
    return model_encoder_tv_input


def build_decoder_withbias(neurons, ic_dim, factors, n_hidden_encode,
                            n_input, n_dynamics, 
                            generator_type='gru', 
                            noise_stddev=0, 
                            logcis=None):
    '''Observation model for neural activity.'''


    IC = tf.keras.Input(shape=(ic_dim), name='Initial condition')
    Input = tf.keras.Input(shape=(None, n_input), name='Input')
    SessionInput = tf.keras.Input(shape=(), dtype=tf.int32, name='SessionInput')

    #
    ic_generators = []
    factors_list = []
    generators = []
    factor_mappings = []
    states = []

    for idynamics in range(n_dynamics):
        ic_generators += [tf.keras.layers.Dense(n_hidden_encode)]

        if noise_stddev == 0:
            if generator_type == 'gru':
                generator = tf.keras.layers.GRU(n_hidden_encode, return_sequences=True)
            elif generator_type == 'linear':
                generator = tf.keras.layers.SimpleRNN(n_hidden_encode, activation=None, return_sequences=True)
            else: 
                raise ValueError(f'generator type: {generator_type} not supported')
        else:
            if generator_type == 'gru':
                generator = tf.keras.layers.RNN(NoisyGRUCell(n_hidden_encode, noise_stddev=noise_stddev), return_sequences=True)
            else:
                raise ValueError(f'generator type: {generator_type} not supported')

        generator_act = generator(Input, initial_state=ic_generators[-1](IC))
        
        factor_mapping = tf.keras.layers.Dense(factors)
        factor = factor_mapping(generator_act)
        
        generators += [generator]
        factors_list += [factor]
        factor_mappings += [factor_mapping]
        states += [generator_act]

    factors_stacked = tf.stack(factors_list)
    states_stacked = tf.stack(states)

    # factors_stacked = tf.einsum('dijk,id->dijk', factors_stacked, DynSelInput)
    states = tf.reduce_sum(states_stacked, 0)
    factor = tf.reduce_sum(factors_stacked, 0)

    # output_mapping = tf.keras.layers.Dense(neurons)
    output_mapping = SwitchLayer(factors, neurons, dropout_rate=0)
    log_firingrate = output_mapping((SessionInput, factor))
    # log_firingrate = output_mapping(factor)
    if logcis is not None:
        log_firingrate += logcis
    firing_rate = tf.exp(log_firingrate)

    decoder_neural = tf.keras.Model(inputs=[Input, IC, SessionInput],
                                            outputs=[firing_rate, factor, factors_stacked, states_stacked])

    return decoder_neural, generators, ic_generators, output_mapping, factor_mappings

########################################################
# Loss functions
def loss_neural(neural_sample, firing_rate_prediction):
    '''Poisson prediction loss'''
    return tf.reduce_sum(firing_rate_prediction - neural_sample * tf.math.log(firing_rate_prediction))
    
    # print('EXPERIMENTAL LOSS - DROPOUT FOR NEURONS!!! ')
    # neurons_mask = tf.keras.layers.Dropout(0.9, 
    #                                              noise_shape=(1, 1, self.neurons))(1 + 0 * neural_sample)

    # return tf.reduce_sum(neurons_mask * (firing_rate_prediction - 
    #                                      neural_sample * tf.math.log(firing_rate_prediction)))

#         return 0.5 * tf.reduce_mean(
#                         ((neural_sample - firing_rate_prediction) ** 2) / (self.params['neural_sigma'] ** 2))
#         return tf.reduce_mean(((neural_sample - firing_rate_prediction) ** 2))

def loss_kl_gauss(x_mu_phi, x_sig_phi):
    '''KL divergence with standard N(0, 1). '''
    # no prior for first term!
    prior_mean = 0
    prior_var = 1
    prior_loss = tf.reduce_sum(0.5 * tf.math.log(prior_var ** 2 / x_sig_phi ** 2) +
                                (x_sig_phi ** 2 + (x_mu_phi - prior_mean) ** 2) / (2 * prior_var ** 2))

    return prior_loss

def loss_smooth_autoregressive(x_smooth, smoothness_alpha):
    # Gaussian smoothness prior
    nll_smooth = tf.reduce_sum(0.5 * (x_smooth[:, 0, :]) ** 2) # standard gaussian loss for t=0
    nll_smooth = nll_smooth + tf.reduce_sum(0.5 * (x_smooth[:, 1:, :] - smoothness_alpha * x_smooth[:, :-1, :]) ** 2)
    return nll_smooth

def loss_ic_prior(x_mu_phi, x_sig_phi, x_sample):
    return loss_kl_gauss(x_mu_phi, x_sig_phi)

def loss_inp_prior(x_mu_phi, x_sig_phi, x_sample):
    return loss_kl_gauss(x_mu_phi, x_sig_phi)

def loss_l2(x):
    return tf.reduce_sum(x ** 2)

def loss_l2_smooth(x):
    return tf.reduce_sum((x[:, 1:, :] - x[:, :-1, :]) ** 2)

def loss_l1(x):
    return tf.reduce_sum(tf.abs(x))

def l2_loss_all_traininable_params(model):
    return tf.reduce_sum([tf.reduce_sum(xx ** 2) for xx in model.trainable_variables])

def temporal_sparsity(x):
    # L1 L2 norm over time
    x = tf.stack(x, 0)
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(x**2, -1)))



########################################################
# Build LFADS models

class LFADS():
    '''Feed forward neural activity observation, LDS dynamics, RNN inference.'''

    def __init__(self, params, controller_class=None, controller_args=None):
        neurons = params['neurons']
        factors = params['factors']
        ic_dim = params['ic_dim']
        inp_dim = params['inp_dim']
        self.inp_dim = inp_dim
        n_sessions = params['n_sessions']
        self.input_exp_mean = params['input_exp_mean']

        n_hidden_decode_ic = params['n_hidden_decode_ic']
        n_hidden_decode_inp = params['n_hidden_decode_inp']
        n_hidden_encode = params['n_hidden_encode']

        self.lams = params['lams']
        self.dropout_rate = params['dropout_rate']

        if controller_class is None:
            self.controller_class = ControllerGeneratorCell
        else:
            self.controller_class = controller_class

        if controller_args is None:
            self.controller_args = (n_hidden_decode_inp, n_hidden_encode,
                                    inp_dim, factors)
        else:
            self.controller_args = controller_args


        self.encoder_ic = build_encoder_ic(neurons, ic_dim, 
                                                              n_hidden_decode_ic, 
                                                              n_sessions, 
                                                              self.dropout_rate)                                                                   

        self.decoder = self.controller_generator(neurons,
                                                                    ic_dim, inp_dim,
                                                                    factors,
                                                                    n_hidden_encode,
                                                                    n_hidden_decode_inp,
                                                                    n_sessions)
        self.trainable_variables = (self.encoder_ic.trainable_variables +
                                    self.decoder.trainable_variables)

        self.if_input = True


    def controller_generator(self, neurons, ic_dim, inp_dim, factors,
                             n_hidden_encode, n_hidden_decode_inp, n_sessions):
        '''Observation model for neural activity.'''

        SessionInput = tf.keras.Input(shape=(n_sessions), name='Session')
        ICInput = tf.keras.Input(shape=(ic_dim), name='Initial condition')
        NeuralInput = tf.keras.Input(shape=(None, neurons), name='neural_input')

        # generate embeddeding for controller
        xInput_transformed = tf.keras.layers.Dropout(self.dropout_rate)(NeuralInput)
        xf = tf.keras.layers.GRU(n_hidden_decode_inp, return_sequences=True)(xInput_transformed)
        xb = tf.keras.layers.GRU(n_hidden_decode_inp, return_sequences=True)(xInput_transformed[:, ::-1, :])
        e = tf.concat([xf[:, :, :], xb[:, :, :]], axis=2)

        #
        ic_generator = tf.keras.layers.Dense(n_hidden_encode)(ICInput)

        initial_state = [
            tf.zeros((tf.shape(NeuralInput)[0],
                      n_hidden_decode_inp)),
            ic_generator,
            tf.zeros((tf.shape(NeuralInput)[0],
                      factors))]

        if self.controller_class == ControllerGeneratorSmoothInputCell:
            initial_state += [tf.zeros((tf.shape(NeuralInput)[0], inp_dim))]

        if self.controller_class == ControllerGeneratorSmoothInputCell2:
            initial_state += [tf.zeros((tf.shape(NeuralInput)[0], inp_dim))]

        ft, it, it_mean, it_sd = tf.keras.layers.RNN(self.controller_class(*self.controller_args),
                                                     return_sequences=True)(e, initial_state=initial_state)

        firing_rate = tf.keras.layers.Dense(neurons)(
            ft)  # + tf.keras.layers.Dense(neurons)(it)  # Add an immediate shortcut from inferred input
        firing_rate = tf.exp(firing_rate)

        model_encoding_neural = tf.keras.Model(inputs=[NeuralInput, ICInput, SessionInput],
                                               outputs=[firing_rate, ft, it, it_mean, it_sd])

        return model_encoding_neural


    def get_loss(self, neural_sample, session_id, kl_scale=1, training=False):
        # get losses
        losses, xk_output = get_loss_components(self, neural_sample, session_id, training=training)

        loss = (self.lams[0] * losses[0] +
                kl_scale * self.lams[1] * losses[1] +
                self.lams[2] * losses[2] +
                self.lams[3] * losses[3] +
                self.lams[4] * losses[4] +
                kl_scale * self.lams[5] * losses[5] +
                self.lams[6] * losses[6] +
                self.lams[7] * losses[7] +
                self.lams[8] * losses[8])

        return loss, xk_output

    def get_factors_inputs_ic(self, neural_sample, session_id, training=False):
        ic_mu, ic_sig2 = self.encoder_ic((neural_sample, session_id), training=training)
        ic_sample = ic_mu + tf.random.normal(ic_mu.shape) * ic_sig2
        # ic_sample = 0 * ic_sample
        # print('!!!!!!!!!!!!!!!Initial condition zeroed out!!!!!!!!!!!!!!!!!!!')

        # neural encoding
        firing_rate_prediction, factors, x_input, x_input_mean, x_input_sd = self.decoder(
            (neural_sample, ic_sample, session_id), training=training)

        return factors, x_input.numpy(), ic_sample.numpy(), x_input_mean.numpy(), x_input_sd.numpy(), firing_rate_prediction.numpy()

    def save(self, filename):
        self.encoder_ic.save_weights(filename + '_encoder_ic')
        self.decoder.save_weights(filename + '_decoder')
        print('Saved')

    def load(self, filename):
        self.encoder_ic.load_weights(filename + '_encoder_ic')
        self.decoder.load_weights(filename + '_decoder')
        print('Loaded')


# class LFADSMultiGeneratorNoInput(LFADS):

#     def __init__(self, params):
#         neurons = params['neurons']
#         self.neurons = neurons
#         factors = params['factors']
#         ic_dim = params['ic_dim']
#         n_sessions = params['n_sessions']

#         n_hidden_decode_ic = params['n_hidden_decode_ic']

#         n_hidden_encode = params['n_hidden_encode']
#         if 'n_dynamics' not in params.keys():
#             n_dynamics = 1
#         else:
#             n_dynamics = params['n_dynamics']
#         self.n_dynamics = n_dynamics

#         self.lams = params['lams']
#         self.dropout_rate = params['dropout_rate']
#         if 'noise_stddev' in params.keys():
#             self.noise_stddev = params['noise_stddev']
#         else:
#             self.noise_stddev = 0

#         self.encoder_ic = self.recognition_network_initial_value(neurons, ic_dim,
#                                                                            n_hidden_decode_ic, n_sessions)

#         self.model_decoder, self.generators, self.ic_to_state = self.controller_generator(neurons,
#                                                                     ic_dim,
#                                                                     factors,
#                                                                     n_hidden_encode,
#                                                                     n_dynamics)
#         self.trainable_variables = (self.encoder_ic.trainable_variables +
#                                     self.model_decoder.trainable_variables)

#         self.if_input = False




#     def controller_generator(self,neurons, ic_dim, factors, n_hidden_encode,
#                              n_dynamics):
#         '''Observation model for neural activity.'''


#         ICInput = tf.keras.Input(shape=(ic_dim), name='Initial condition')
#         NeuralInput = tf.keras.Input(shape=(None, neurons), name='neural_input')

#         #
#         ic_generators = []
#         generators = []
#         factors_list = []
#         for idynamics in range(n_dynamics):
#             ic_generators += [tf.keras.layers.Dense(n_hidden_encode)]
            
#             # generator = tf.keras.layers.GRU(n_hidden_encode, return_sequences=True)
#             generator = tf.keras.layers.RNN(NoisyGRUCell(n_hidden_encode, noise_stddev=self.noise_stddev), return_sequences=True)
#             generator_act = generator(0 * NeuralInput, initial_state=ic_generators[-1](ICInput))
#             factor = tf.keras.layers.Dense(factors)(generator_act)
            
#             factors_list += [factor]
#             generators += [generator]

#         factors_stacked = tf.stack(factors_list)
#         factor = tf.reduce_sum(tf.stack(factors_list), 0)
#         firing_rate = tf.exp(tf.keras.layers.Dense(neurons)(factor))

#         model_encoding_neural = tf.keras.Model(inputs=[NeuralInput, ICInput,],
#                                                outputs=[firing_rate,  factor, factors_stacked])

#         return model_encoding_neural, generators, ic_generators

#     def get_factors_ic(self, neural_sample, session_id, training=False):
#         ic_mu, ic_sig2 = self.encoder_ic((neural_sample, session_id), training=training)
#         ic_sample = ic_mu + tf.random.normal(ic_mu.shape) * ic_sig2

#         # neural encoding
#         op = self.model_decoder(
#             (neural_sample, ic_sample), training=training)
#         firing_rate_prediction, factors, factors_stacked = op[0], op[1], op[2]

#         return factors.numpy(), ic_sample.numpy(), firing_rate_prediction.numpy(), factors_stacked.numpy()

#     def save(self, filename):
#         super().save(filename) # from LFADS

#     def load(self, filename):
#         super().load(filename)

#     def get_loss(self, neural_sample, session_id, kl_scale=1, training=True):
#         # get losses
#         losses, xk_output = get_loss_components2(self, neural_sample, session_id, training=training)

#         loss = (self.lams[0] * losses[0] +
#                 kl_scale * self.lams[1] * losses[1] +
#                 self.lams[2] * losses[2] +
#                 self.lams[3] * losses[3] +
#                 self.lams[4] * losses[4] +
#                 kl_scale * self.lams[5] * losses[5]
#                )

#         return loss, xk_output



class LFADSMultiGeneratorBias():

    def __init__(self, neurons, factors, 
                ic_dim, n_sessions, n_hidden_encode_ic, n_hidden_encode_bias, n_hidden_decode, lams, bias_dim=1,
                dropout_rate=0, n_dynamics=1, logcis=None, noise_stddev=0, generator_type='gru', 
                use_bias=True, n_hidden_encode_tv_input=100, tv_input_dim=0, use_tv_input=False, tv_input_smoothness_alpha=0): #params):

        self.neurons = neurons
        self.n_dynamics = n_dynamics
        self.lams = lams
        self.dropout_rate = dropout_rate
        self.logcis = logcis
        self.noise_stddev = noise_stddev
        self.generator_type = generator_type
        self.use_bias = use_bias
        self.bias_dim = bias_dim
        self.tv_input_dim = tv_input_dim
        self.use_tv_input = use_tv_input
        self.tv_input_smoothness_alpha = tv_input_smoothness_alpha

        
        self.encoder_ic = build_encoder_ic(neurons, ic_dim, n_hidden_encode_ic, 
                                           n_sessions, dropout_rate)    
                                           
        op = build_decoder_withbias(neurons, ic_dim, factors, n_hidden_decode,
                                    bias_dim + tv_input_dim, n_dynamics, 
                                    generator_type='gru', 
                                    noise_stddev=0, 
                                    logcis=None)
        self.decoder, self.generators, self.ic_to_state, self.factor_op_mapping, self.factor_mappings = op

        self.trainable_variables = (self.encoder_ic.trainable_variables +
                                    self.decoder.trainable_variables)

        if use_bias:
            self.encoder_bias = build_encoder_bias(neurons, bias_dim, n_hidden_encode_bias, dropout_rate)
            self.trainable_variables += self.encoder_bias.trainable_variables
        
        if use_tv_input:
            self.encoder_tv_input = build_encoder_tv_input(neurons, self.tv_input_dim, n_hidden_encode_tv_input, dropout_rate)
            self.trainable_variables += self.encoder_tv_input.trainable_variables
        

    def run(self, neural_sample, session_id, training=False, return_numpy=False):

        # encode initial condition
        ic_mu, ic_sig = self.encoder_ic((neural_sample, session_id), training=training)
        ic_sample = ic_mu + tf.random.normal(ic_mu.shape) * ic_sig

        # encode bias
        if self.use_bias:
            bias_mu, bias_sig = self.encoder_bias((neural_sample, session_id), training=training)
            bias_sample = bias_mu + tf.random.normal(bias_mu.shape) * bias_sig
        else:
            bias_sample = 0 * neural_sample[:, 0, :1] + tf.zeros((self.bias_dim))  # No bias used
        
        bias_time_sample = (tf.expand_dims(0 * neural_sample[:, :, 0], 2) + tf.expand_dims(bias_sample, 1))
        
        
        # encode time-varying input
        if self.use_tv_input:
            tv_input_mu, tv_input_sig = self.encoder_tv_input((neural_sample, session_id), training=training)
            tv_input_sample = tv_input_mu + tf.random.normal(tv_input_mu.shape) * tv_input_sig
        else:
            tv_input_sample = 0 * tf.expand_dims(neural_sample[:, :, 0], 2) + tf.zeros((self.tv_input_dim))

        # combine
        all_inputs_sample = tf.concat([bias_time_sample, tv_input_sample], axis=2)


        # decode
        op = self.decoder((all_inputs_sample, ic_sample, session_id), training=training)
        firing_rate_prediction, factors, factors_stacked, states_stacked = op[0], op[1], op[2], op[3]

        return_ = {
                   'ic_sample': ic_sample, 'ic_mu': ic_mu, 'ic_sig': ic_sig, 
                   'firing_rate': firing_rate_prediction, 'factors': factors, 
                   'factors_stacked': factors_stacked, 'states_stacked': states_stacked
                   }

        if self.use_bias:
            return_.update({'bias_sample': bias_sample, 'bias_mu': bias_mu, 'bias_sig': bias_sig})
        
        if self.use_tv_input:
            return_.update({'tv_input_sample': tv_input_sample, 'tv_input_mu': tv_input_mu, 'tv_input_sig': tv_input_sig})

        if return_numpy:
            for k in return_.keys():
                return_[k] = return_[k].numpy()

        return return_

    def save(self, filename, suffix=''):
        self.encoder_ic.save_weights(filename + '_encoder_ic' + suffix)
        
        if self.use_bias:
            self.encoder_bias.save_weights(filename + '_encoder_bias' + suffix)

        if self.use_tv_input:
            self.encoder_tv_input.save_weights(filename + '_encoder_tv_input' + suffix)
        
        self.decoder.save_weights(filename + '_decoder' + suffix)

    def load(self, filename, suffix=''):
        self.encoder_ic.load_weights(filename + '_encoder_ic' + suffix)
        
        if self.use_bias:
            self.encoder_bias.load_weights(filename + '_encoder_bias' + suffix)
        
        if self.use_tv_input:
            self.encoder_tv_input.load_weights(filename + '_encoder_tv_input' + suffix)

        self.decoder.load_weights(filename + '_decoder' + suffix)

    def get_loss(self, neural_sample, session_id, kl_scale=[1, 1], training=True):
        # get losses
        losses, xk_output = get_loss_components(self, neural_sample, session_id, training=training)

        loss = (self.lams[0] * losses[0] +
                kl_scale[0] * self.lams[1] * losses[1] +
                self.lams[2] * losses[2] +
                self.lams[3] * losses[3] +
                kl_scale[0] * self.lams[4] * losses[4] +
                self.lams[5] * losses[5] +
                kl_scale[0] * self.lams[6] * losses[6] + 
                kl_scale[0] * self.lams[7] * losses[7]
               )

        return loss, xk_output


    def get_elbo(self, neural_sample, session_id, training=True, n_resamples=1):
        # get losses
        elbo_log = []
        losses_log = []
        for iresample in range(n_resamples):
            losses, xk_output = get_loss_components(self, neural_sample, session_id, training=training)

            elbo = (losses[0] +   # the neural prediction loss
                    losses[1])   # the KL loss on initial condition   
            
            if self.use_bias:
                elbo += losses[6]

            if self.use_tv_input:
                elbo += losses[7]

            losses_log += [losses]
            elbo_log += [elbo]

        return np.mean(elbo_log), np.array(losses_log).mean(0)

    # def get_factors_ic(self, neural_sample, session_id, get_inputs=False, training=False, bias_scale=1):
    #     # get intial condition
    #     ic_mu, ic_sig2 = self.encoder_ic((neural_sample, session_id), training=training)
    #     ic_sample = ic_mu + tf.random.normal(ic_mu.shape) * ic_sig2

    #     # select dynamics
    #     inp_mu, inp_sig = self.encoder_bias(neural_sample, training=training)
    #     inp_sample = inp_mu + tf.random.normal(inp_mu.shape) * inp_sig
    #     inp_sample = bias_scale * inp_sample

    #     # neural encoding
    #     inp_time = tf.expand_dims(0 * neural_sample[:, :, 0], 2) +tf.expand_dims(inp_sample, 1)
    #     #print('inp_time', inp_time.shape)
    #     op = self.decoder((inp_time, ic_sample), training=training)
    #     firing_rate_prediction, factors, factors_stacked = op[0], op[1], op[2]

    #     if not get_inputs:
    #         return factors.numpy(), ic_sample.numpy(), firing_rate_prediction.numpy(), factors_stacked.numpy()
    #     else:
    #         return factors.numpy(), ic_sample.numpy(), firing_rate_prediction.numpy(), factors_stacked.numpy(), inp_sample.numpy()


# @tf.function
def get_loss_components(model, neural_sample, session_id, training=True):

    output = model.run(neural_sample, session_id, training=training)
     
    # get losses
    losses = []
    losses += [loss_neural(neural_sample, output['firing_rate'])]
    losses += [loss_kl_gauss(output['ic_mu'], output['ic_sig'])] #[loss_ic_prior(output['ic_mu'], output['ic_sig'], output['ic_sample'])]
    losses += [loss_l2(output['factors'])]
    losses += [loss_l2_smooth(output['factors'])]
    losses += [l2_loss_all_traininable_params(model)]
    losses += [temporal_sparsity(output['factors_stacked'])]
    if model.use_bias:
        losses += [loss_kl_gauss(output['bias_mu'], output['bias_sig'])] #[loss_ic_prior(output['bias_mu'], output['bias_sig'], output['bias_sample'])]
    else:
        losses += [tf.constant(np.array(0).astype(np.float32))]

    if model.use_tv_input:
        
        if model.tv_input_smoothness_alpha > 0:
            # smoothness prior
            losses += [loss_smooth_autoregressive(output['tv_input_sample'], model.tv_input_smoothness_alpha)]
        else:
            # no smoothness prior
            losses += [loss_kl_gauss(output['tv_input_mu'], output['tv_input_sig'])] 
        # #[loss_ic_prior(output['tv_input_mu'], output['tv_input_sig'], output['tv_input_sample'])]
    else:
        losses += [tf.constant(np.array(0).astype(np.float32))]

    output.update({'loss_neural_prediction': losses[0], 'losses': losses})
    return losses, output




def train(ds_series, model_fn, data_test=None,
          lr_init=0.001, lr_stop=1e-6, lams=None, n_steps=5000, to_plot=False, 
          kl_warmup_start=0, kl_warmup_end=0, # bias_warmup_start=0, bias_warmup_end=0,
          decay_factor=0.9999, gradient_clipping_norm=10, 
          savefile=None, n_eval_samples=None, patience_till_lr_decay=5, 
          save_freq=None, data_val_weight=None):  # n_steps_improvement_for_decay=5,

        
    import time
    lr = lr_init
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if data_val_weight is None:
        data_val_weight = np.ones(len(data_test))

    def train_step(n_sample, session_sample, kl_scale):
        with tf.GradientTape() as tape:
            print('train_step', n_sample.shape, session_sample.shape, session_sample, 'learning rate', lr)
            loss, x_output = model_fn.get_loss(n_sample, session_sample, kl_scale, training=True)

            gradients = tape.gradient(loss, model_fn.trainable_variables)
            gradients = [tf.clip_by_norm(g, gradient_clipping_norm) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model_fn.trainable_variables))
        return loss, x_output

    def eval_step(n_sample, session_sample, kl_scale):
        with tf.GradientTape() as tape:
            print('eval step', n_sample.shape, session_sample.shape)
            loss, x_output = model_fn.get_loss(n_sample, session_sample, kl_scale, training=False)
        return loss, x_output

    def test_dataset(data_test):

        datay_test, session_test = data_test['neural'], data_test['session']

        # n_test = np.random.choice(datay_test.shape[0], 100)
        n_test = np.arange(datay_test.shape[0])
        if n_eval_samples is not None:
            test_ex = np.random.choice(datay_test.shape[0], n_eval_samples)
            datay_use, sessiony_use = datay_test[test_ex, ...], session_test[test_ex, ...]
        else:
            datay_use, sessiony_use = datay_test, session_test

        loss_eval, k_pred_eval = eval_step(datay_use, tf.constant(sessiony_use[0, 0].astype(np.int32)), kl_scale)  # have KL scale =1 no matter what.
        
        return loss_eval, k_pred_eval

    loss_np = []
    loss_min_np = []
    loss_test_np = []
    loss_test_min = np.inf
    loss_test_min_prebias = np.inf
    loss_test_pred_np = []
    all_losses = []

    iiter = 0
    n_steps_since_improvement = 0
    loss_np_prev = np.inf
    start_time = time.time()
    for n_sample, session_sample in ds_series.repeat().take(n_steps):
        print(iiter)
        
        if lr < lr_stop:
            break

        iiter += 1

        kl_scale = []
        for warm_start, warm_end in zip(kl_warmup_start, kl_warmup_end):
            if iiter < warm_start:
                kl_scale += [0]
            elif iiter > warm_end:
                kl_scale += [1]
            else:
                kl_scale += [(iiter - warm_start) / (warm_end - warm_start)]

        loss, k_pred = train_step(n_sample, session_sample[0, 0], kl_scale)
        # print(np.sum(np.isnan(k_pred['x_input'].numpy())))
        # print(model_fn.smoothness_alpha_use, model_fn.sparsity_nu_use)
        # print('k_pred losses', k_pred['losses'])
        all_losses += [[l.numpy() for l in k_pred['losses']]]

        # print(loss, 'kl_scale', kl_scale, loss_np_prev, all_losses[-1])

        loss_np += [loss.numpy()]
        loss_min_np += [np.min(loss_np)]

        if 'exec_time' not in locals():
            exec_time = time.time() - start_time
        else:
            exec_time = 0.9 * exec_time + 0.1 * (time.time() - start_time)
        start_time = time.time()

        if save_freq is not None:
            if (iiter % save_freq == 0):
                model_fn.save(savefile, suffix='-iter=' + str(iiter))

        if (data_test is not None) and (iiter % 10 == 1):
            
            loss_eval = 0
            loss_eval_pred = 0
            from tqdm import tqdm
            # from IPython import embed; embed()
            id = -1
            for d in tqdm(data_test):
                id += 1
                loss_eval_, k_pred_eval = test_dataset(d)
                loss_eval += data_val_weight[id] * loss_eval_.numpy()
                print('eval', data_val_weight[id], loss_eval_.numpy())
                loss_eval_pred +=  data_val_weight[id] * k_pred_eval['loss_neural_prediction']

            loss_test_np += [loss_eval]
            loss_test_pred_np += [loss_eval_pred]

            # datay_test, session_test = data_test['neural'], data_test['session']


            # n_test = np.random.choice(datay_test.shape[0], 100)
            # n_test = np.arange(datay_test.shape[0])
            # if n_eval_samples is not None:
            #     test_ex = np.random.choice(datay_test.shape[0], n_eval_samples)
            #     datay_use, sessiony_use = datay_test[test_ex, ...], session_test[test_ex, ...]
            # else:
            #     datay_use, sessiony_use = datay_test, session_test

            # loss_eval, k_pred_eval = eval_step(datay_use, tf.constant(sessiony_use[0, 0].astype(np.int32)), kl_scale)  # have KL scale =1 no matter what.

            # loss_test_np += [loss_eval.numpy()]
            # loss_test_pred_np += [k_pred_eval['loss_neural_prediction']]
            print('loss_test_min', loss_test_min, 'loss_test_min_prebias', loss_test_min_prebias, 'current_loss', loss_test_np[-1])

            
            if (loss_test_np[-1] < loss_test_min) and (iiter > np.max(kl_warmup_end)) :
                loss_test_min = loss_test_np[-1]

                n_steps_since_improvement = 0
                if savefile is not None :
                    print('Saved model')
                    model_fn.save(savefile)

            if (loss_test_np[-1] > loss_test_min) and (iiter > np.max(kl_warmup_end)):
                n_steps_since_improvement += 1
                print('n_steps_since_improvement', n_steps_since_improvement)
                if n_steps_since_improvement > patience_till_lr_decay:
                    n_steps_since_improvement = 0
                    lr = lr * decay_factor
                    print('learning rate decayed', lr)
                    optimizer.lr.assign(lr)


        if iiter % 100 == 1:

            print(f'Step: {iiter} Loss: {loss_np[-1]}, Avg exec time: {exec_time}s')

            #                 print('.', end='')
            if not to_plot:
                continue

            from IPython import display
            from IPython.display import clear_output
            clear_output(wait=True)

            plt.figure(figsize=(30, 5))
            all_losses_np = np.array(all_losses)
            for iloss_plot in range(all_losses_np.shape[-1]):
                plt.subplot(3, 5, iloss_plot + 1)
                if model_fn.lams[iloss_plot] != 0:
                    plt.plot(all_losses_np[:, iloss_plot])

            plt.figure(figsize=(30, 5))
            plt.subplot(2, 7, 1)
            plt.plot(loss_np[100:])
            plt.plot(loss_min_np[100:])
            plt.title('Train')


            plt.subplot(2, 7, 5)
            plt.plot(k_pred['factors'][0, ...])
            plt.title('factors')

            '''
            plt.subplot(2, 7, 6)
            plt.plot(k_pred['ic_sample'][:, 0], k_pred['ic_sample'][:, 1], '.')
            plt.xlim([-4, 4])
            plt.ylim([-4, 4])
            plt.plot(0, 0, 'r+')
            '''
            plt.subplot(2, 7, 6)
            plt.plot(k_pred['ic_sample'])
            plt.title('initial condition (training)')
            
            if 'inp_sample' in k_pred.keys():
                plt.subplot(2, 7, 7)
                plt.plot(k_pred['inp_sample'])
                plt.title('input sample (training)')

            plt.suptitle(f'Loss:{loss_np[-1]}, \n Exec_time {exec_time}')

            if data_test is not None:

                plt.subplot(2, 7, 8)
                plt.plot(loss_test_np)
                plt.plot(loss_test_pred_np)
                # plt.ylim([np.nanmin(loss_test_np), np.percentile(loss_test_np, 99) * 1.1])
                plt.title('Eval neural prediction loss')

                plt.subplot(2, 7, 9)
                plt.plot(loss_test_np[20:])
                plt.plot(loss_test_pred_np[20:])
                # plt.ylim([np.nanmin(loss_test_np), np.percentile(loss_test_np, 99) * 1.1])
                plt.title('Eval neural prediction loss (200th step onwards)')

                '''
                # Eval plots, turned off for now.
                print('Eval')
                print('.', end='')
                op = model_fn.run(datay_test,
                                  tf.constant(session_test[0, 0].astype(np.int32)),
                                  training=False, 
                                  return_numpy=True)
                factors_, ic_sample, firing_rate, factors_stacked = op['factors'], op['ic_sample'], op['firing_rate'], op['factors_stacked']                                                                 

                plt.subplot(2, 7, 12)
                iex = np.random.choice(op['factors'].shape[0])
                plt.plot(op['factors'][iex, :, :])
                plt.title('Eval factors')
                
                            
                plt.subplot(2, 7, 13)
                plt.plot(k_pred_eval['ic_sample'])
                plt.title('initial condition (eval)')
                
                if 'inp_sample' in k_pred_eval.keys():
                    plt.subplot(2, 7, 14)
                    plt.plot(k_pred_eval['inp_sample'])
                    plt.title('input sample (eval)')

                '''

                '''
                # useful for dropout models
                # model_fn.analysis(datay_test[n_test, ...])
                analysis_mpca(data_test['delays'], data_test['cues'], firing_rate)
                plt.suptitle('Firing rate')

                analysis_mpca(data_test['delays'], data_test['cues'], factors_)
                plt.suptitle('Factors')

                for idynamics in range(factors_stacked.shape[0]):
                    analysis_mpca(data_test['delays'], data_test['cues'], factors_stacked[idynamics, ...])
                    plt.suptitle(f'Factors dynamics: {idynamics}')
                '''

            plt.show()
    return loss_np

'''
def load_model_from_config(config, sweep_dir, config_file, n_channels):
    # Fit model
    params = {}
    params.update({'neurons': n_channels})
    params.update({'factors': config['model']['factors']}) 
    params.update({'ic_dim': config['model']['ic_dim']}) #128}) 
    n_sessions = 1
    params.update({'n_sessions': n_sessions})

    params.update({'n_hidden_decode_ic': config['model']['n_hidden_decode_ic']})
    params.update({'n_layers_decode': config['model']['n_layers_decode']})
    params.update({'n_hidden_encode': config['model']['n_hidden_encode']}) #128})
    params.update({'n_layers_encode': config['model']['n_layers_encode']})
    params.update({'input_exp_mean': config['model']['input_exp_mean']})
    params.update({'dropout_rate': config['model']['dropout_rate']}) #0.02, 0.1??}) ## make sure this parameter is being used
    # coordinated dropout

    params.update({'kl_warmup_start': np.array([config['model']['kl_warmup_start_ic'], 
                                                config['model']['kl_warmup_start_bias']]), 
                   'kl_warmup_end': np.array([config['model']['kl_warmup_start_ic'] + config['model']['kl_warmup_duration_ic'], 
                                              config['model']['kl_warmup_start_bias'] + config['model']['kl_warmup_duration_bias']])}) # warmup for I.C. and bias respsectively.
    params.update({'bias_warmup_start': config['model']['bias_warmup_start'], 
                    'bias_warmup_end': config['model']['bias_warmup_start'] + config['model']['bias_warmup_duration']})

    params.update({'lams': np.array([config['model']['lam_predict'], 
                                    config['model']['lam_kl_ic'], 
                                    config['model']['lam_l2'], 
                                    config['model']['lam_l2'], 
                                    config['model']['lam_l2'], 
                                    0,
                                    config['model']['lam_kl_bias']])})

    # LFADS with multiple generators, and smoothness
    params.update({'n_dynamics': config['model']['n_dynamics']})
    params.update({'n_hidden_decode_input': config['model']['n_hidden_decode_input']})
    params.update({'n_inputs': config['model']['n_inputs']}) #10 or 128})
    params.update({'noise_stddev': config['model']['noise_stddev'] / np.sqrt(config['model']['n_hidden_encode'])})
    
    params.update({'generator_type': config['model']['generator_type']})
    
    model = LFADSMultiGeneratorConstantInput(params)
    model.load(f'{sweep_dir}/{config_file}/model')
    
    model_nobias = LFADSMultiGeneratorConstantInput(params)
    model_nobias.load(f'{sweep_dir}/{config_file}/modelnobias')

    results = pickle.load(open(f'{sweep_dir}/{config_file}/results.pkl', 'rb'))
    
    return model, model_nobias, results
'''

'''
def load_model_from_config(config, sweep_dir, config_file, n_channels):
    params = {}
    params.update({'neurons': n_channels})
    params.update({'factors': config['model']['factors']}) 
    params.update({'ic_dim': config['model']['ic_dim']})
    params.update({'n_sessions': 1})

    params.update({'n_hidden_encode_ic': config['model']['n_hidden_encode_ic']})
    params.update({'n_hidden_decode': config['model']['n_hidden_decode']})
    params.update({'dropout_rate': config['model']['dropout_rate']}) 

    params.update({'lams': np.array([config['model']['lam_predict'], 
                                    config['model']['lam_kl_ic'], 
                                    0, 
                                    0, 
                                    config['model']['lam_l2'], 
                                    0,
                                    config['model']['lam_kl_bias']])})



    # # LFADS with multiple generators, and smoothness
    params.update({'n_dynamics': config['model']['n_dynamics']})
    params.update({'n_hidden_encode_bias': config['model']['n_hidden_encode_bias']})
    params.update({'bias_dim': config['model']['bias_dim']}) 
    params.update({'noise_stddev': config['model']['noise_stddev'] / np.sqrt(config['model']['n_hidden_decode'])}) 

    params.update({'generator_type':  config['model']['generator_type']})

    params.update({'use_bias': config['model']['use_bias']})
    model = LFADSMultiGeneratorBias(**params)

    model.load(f'{sweep_dir}/{config_file}/model')
    return model
'''

'''
def load_model_from_config(config, model_filename, n_channels):
    params = {}
    params.update({'neurons': n_channels}) #data['spikes_trials'][0].shape[-1]})
    params.update({'factors': config['model']['factors']}) 
    params.update({'ic_dim': config['model']['ic_dim']})
    params.update({'n_sessions': 1})

    params.update({'n_hidden_encode_ic': config['model']['n_hidden_encode_ic']})
    params.update({'n_hidden_decode': config['model']['n_hidden_decode']})
    params.update({'dropout_rate': config['model']['dropout_rate']}) 

    params.update({'lams': np.array([config['model']['lam_predict'], 
                                    config['model']['lam_kl_ic'], 
                                    0, 
                                    0, 
                                    config['model']['lam_l2'], 
                                    0,
                                    config['model']['lam_kl_bias'], 
                                    config['model']['lam_kl_tv_inp']])})



    # # LFADS with multiple generators, and smoothness
    params.update({'n_dynamics': config['model']['n_dynamics']})
    params.update({'n_hidden_encode_bias': config['model']['n_hidden_encode_bias']})
    params.update({'bias_dim': config['model']['bias_dim']}) 
    params.update({'noise_stddev': config['model']['noise_stddev'] / np.sqrt(config['model']['n_hidden_decode'])}) 

    params.update({'generator_type':  config['model']['generator_type']})

    params.update({'use_bias': config['model']['use_bias']})

    params.update({'n_hidden_encode_tv_input': config['model']['n_hidden_encode_tv_input']})
    params.update({'tv_input_dim': config['model']['tv_input_dim']})
    params.update({'tv_input_smoothness_alpha': config['model']['tv_input_smoothness_alpha']})
    params.update({'use_tv_input': config['model']['use_tv_input']})

    model = LFADSMultiGeneratorBias(**params)

    model.load(model_filename)
    return model

'''


def load_model_from_config(config, n_channels, model_filename='', suffix=''):
    params = {}
    params.update({'neurons': n_channels})  #data['spikes_trials'][0].shape[-1]})
    params.update({'factors': config['model']['factors']}) 
    params.update({'ic_dim': config['model']['ic_dim']})
    params.update({'n_sessions': 1})

    params.update({'n_hidden_encode_ic': config['model']['n_hidden_encode_ic']})
    params.update({'n_hidden_decode': config['model']['n_hidden_decode']})
    params.update({'dropout_rate': config['model']['dropout_rate']}) 

    params.update({'lams': np.array([config['model']['lam_predict'], 
                                    config['model']['lam_kl_ic'], 
                                    0, 
                                    0, 
                                    config['model']['lam_l2'], 
                                    0,
                                    config['model']['lam_kl_bias'], 
                                    config['model']['lam_kl_tv_inp']])})



    # # LFADS with multiple generators, and smoothness
    params.update({'n_dynamics': config['model']['n_dynamics']})
    params.update({'n_hidden_encode_bias': config['model']['n_hidden_encode_bias']})
    params.update({'bias_dim': config['model']['bias_dim']}) 
    params.update({'noise_stddev': config['model']['noise_stddev'] / np.sqrt(config['model']['n_hidden_decode'])}) 

    params.update({'generator_type':  config['model']['generator_type']})

    params.update({'use_bias': config['model']['use_bias']})

    params.update({'n_hidden_encode_tv_input': config['model']['n_hidden_encode_tv_input']})
    params.update({'tv_input_dim': config['model']['tv_input_dim']})
    params.update({'tv_input_smoothness_alpha': config['model']['tv_input_smoothness_alpha']})
    params.update({'use_tv_input': config['model']['use_tv_input']})

    model = LFADSMultiGeneratorBias(**params)

    try:
        model.load(model_filename, suffix)
        print('Model loaded')
    except:
        print('Model could not be loaded')

    return model