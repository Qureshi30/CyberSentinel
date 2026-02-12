# Training Data Directory

Place your **KDDTrain+.csv** file here before training the model.

## Where to get the NSL-KDD dataset:

1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download the NSL-KDD dataset
3. Extract the **KDDTrain+.csv** file
4. Place it in this directory

## Expected Format:

The CSV file should contain network traffic features with multiple columns representing different aspects of network connections, such as:

- duration
- protocol_type
- service
- flag
- src_bytes
- dst_bytes
- land
- wrong_fragment
- urgent
- ... (and many more features)
- **label** (the last column containing attack types or "normal")

The training script will automatically convert the labels to binary classification:
- `normal` → **Harmless**
- Any other value → **Threat**
