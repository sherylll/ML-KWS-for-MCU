basic_lstm 128, 16 mfccs
original: 
best validation accuracy is 93.59%, test accuracy = 92.74%

python quant_test.py --checkpoint quant_basic_lstm_128/ckpts/basic_lstm_9347.ckpt-2000
val: 93.30, test: 93.03

4-layer: total no. params = 470540
94.09 -> 94.03  

lstm w projection 128/64
original
best validation accuracy is 93.41%, test accuracy = 93.11%
quant. 8-bit, gate range = 8, input/logits range = 16
(retrain) 93.25, 92.82
// quant. 8-bit, gate range = 8, input/logits range = 16, projection bits = 12
// 93.05, 92.86 -> (retrain) 93.23, 92.70
    
    quant. 12 bits / 16 bits: 93.23, 92.88 / 93.25, 92.72

lstm w projection 128/64 2-layer
best validation accuracy is 94.69%, test accuracy = 94.15
after retrain: 94.40, 94.17


crnn: val-95.5, test-94.6 -> retrain 94.7