basic_lstm 128, 16 mfccs
original: 
best validation accuracy is 93.59%, test accuracy = 92.74%
quant. ap_fixed<8,5>
validation acc. 93.00%, test: 92.45% -> 93.23, 92.90 after retrain

lstm w projection 128/64
original
best validation accuracy is 93.41%, test accuracy = 93.11%
quant. 8-bit, gate range = 8, input/logits range = 16
92.91, 92.64 -> (retrain) 93.25, 92.62
// quant. 8-bit, gate range = 8, input/logits range = 16, projection bits = 12
// 93.05, 92.86 -> (retrain) 93.23, 92.70
    
    quant. 12 bits / 16 bits: 93.23, 92.88 / 93.25, 92.72

lstm w projection 128/64 2-layer
best validation accuracy is 94.69%, test accuracy = 94.15
after retrain: 94.47, 93.97
