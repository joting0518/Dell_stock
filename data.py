"""
plt.plot(df.index[time_steps:], scaler.inverse_transform(scaled_data[time_steps:, 3].reshape(-1, 1)), label='True')
plt.plot(df.index[time_steps:], scaler.inverse_transform(predictions), label='Predicted')
plt.legend()
plt.show()
"""