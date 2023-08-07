# Iowa_Liquor_Sales_Forcasting_LSTM
## Why LSTM for Time Series Forecasting?
Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), are designed to recognize patterns over sequences of data. 
This makes them especially effective for time series forecasting tasks, like predicting liquor sales based on historical data. 
Their architecture allows them to remember patterns over long sequences, making them less susceptible to the vanishing gradient problem compared to traditional RNNs.

## Long Short-Term Memory (LSTM) Networks: A Mathematical Deep Dive

### Basic Concept:

At the core, LSTMs maintain a cell state, as well as a hidden state, across the sequences. They utilize three gates:

- **Forget Gate**: Decides what information from the cell state should be thrown away or kept.
- **Input Gate**: Updates the cell state with new information.
- **Output Gate**: Determines the next hidden state.

### The Math:

Given an input sequence \( X = (x_1, x_2, ..., x_t) \), the LSTM cell computes the cell state \( C_t \) and the hidden state \( h_t \) for each element in the sequence. Here's a breakdown of the calculations:

1. **Forget Gate**:
\[ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \]
Where:
    - \( f_t \) is the forget gate's output
    - \( W_f \) are its weights
    - \( b_f \) is its bias
    - \( \sigma \) is the sigmoid function
    - \( h_{t-1} \) is the previous hidden state
    - \( x_t \) is the current input

2. **Input Gate**:
\[ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \]
\[ \tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \]

3. **Cell State Update**:
\[ C_t = f_t \times C_{t-1} + i_t \times \tilde{C_t} \]

4. **Output Gate**:
\[ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \]
\[ h_t = o_t \times \tanh(C_t) \]

In summary, LSTMs are designed to selectively remember patterns over long durations and are less susceptible to the vanishing gradient problem. The gates within the LSTM help it decide what information to throw away or keep, making it effective for sequence prediction tasks.

## Data Loading and Preprocessing
1. Data Scaling: Neural networks, including LSTMs, perform better when input features are scaled. 
   Typically, data is scaled to a range, like [0, 1], using MinMax scaling or standardized to have a mean of 0 and a standard deviation of 1.

2. Sequence Creation: For time series forecasting with LSTMs, data needs to be transformed into a sequence format where each input sequence has a corresponding output.
   For example, using sales data from the past seven days to predict sales on the eighth day.

## LSTM Model Configuration and Training
1. Model Architecture: The LSTM network typically consists of an input layer followed by one or more LSTM layers.
   A dense layer is then used to produce the forecast. The depth and complexity of the model can vary based on the problem.

2. Loss Function and Optimizer: Mean Squared Error (MSE) is a commonly used loss function for regression problems like forecasting.
   Optimizers like Adam or RMSprop are typically used to minimize this loss over the training epochs.

3. Callbacks and Early Stopping: To prevent overfitting, early stopping can be used.
   This stops training once the model's performance stops improving on a held-out validation dataset.

4. Batch Training: LSTMs can be sensitive to batch size. 
   The batch size defines the number of patterns the model is exposed to before the weights are updated. It can impact training speed and model stability.

## Evaluation and Results
1. Model Predictions: Once the model is trained, it can be used to make predictions on test data.

2. Inverse Scaling: If data was scaled before training, predictions need to be inverse scaled to bring them back to the original scale.

3. Performance Metrics: Metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and others can be used to quantify the model's forecasting accuracy on test data.
