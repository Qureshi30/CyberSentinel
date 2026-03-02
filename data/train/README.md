# Training Data Directory

Place your **KDDTrain+.csv** file here before training the model.

## Where to get the NSL-KDD dataset:

1. Visit: https://www.unb.ca/cic/datasets/nsl.html
2. Download the NSL-KDD dataset
3. Extract the **KDDTrain+.csv** file
4. Place it in this directory

## NSL-KDD Dataset Structure:

Each row represents **one network connection (session)** with features grouped into:

### 1) Basic Connection Features
- **duration**: Connection length in seconds
- **protocol_type**: tcp, udp, icmp
- **service**: http, ftp, telnet, smtp, etc.
- **flag**: Connection status (SF, S0, REJ, etc.)
- **src_bytes**: Bytes sent from source to destination
- **dst_bytes**: Bytes sent from destination to source

### 2) Content-Based Features
- **land**: 1 if source and destination IP/port are same (suspicious)
- **wrong_fragment**: Number of incorrect IP fragments
- **urgent**: Number of urgent packets
- **hot**: Number of sensitive operations
- **num_failed_logins**: Failed login attempts count
- **logged_in**: 1 if login successful
- **num_compromised**: Number of compromised conditions
- **root_shell**: 1 if root shell obtained (attack indicator)
- **su_attempted**: 1 if "su root" attempted
- **num_root**: Number of root accesses
- **num_file_creations**: Files created during connection
- **num_shells**: Shell prompts opened
- **num_access_files**: Access count to sensitive files
- **is_host_login**: 1 if login belongs to host list
- **is_guest_login**: 1 if guest login

### 3) Time-Based Traffic Features (last 2 seconds)
- **count**: Connections to same host
- **srv_count**: Connections to same service
- **serror_rate**: Percentage of SYN errors (0.0-1.0)
- **srv_serror_rate**: SYN error rate for same service
- **rerror_rate**: Percentage of REJ errors (0.0-1.0)
- **srv_rerror_rate**: REJ error rate for same service
- **same_srv_rate**: Fraction of same service connections
- **diff_srv_rate**: Fraction of different service connections
- **srv_diff_host_rate**: Same service, different hosts rate

### 4) Host-Based Traffic Features (last 100 connections)
- **dst_host_count**: Connections to same destination host
- **dst_host_srv_count**: Same service connections on host
- **dst_host_same_srv_rate**: Same service connection fraction
- **dst_host_diff_srv_rate**: Different services fraction
- **dst_host_same_src_port_rate**: Same source port usage rate
- **dst_host_srv_diff_host_rate**: Same service, different hosts rate
- **dst_host_serror_rate**: SYN error rate for destination host
- **dst_host_srv_serror_rate**: SYN error rate for destination service
- **dst_host_rerror_rate**: REJ error rate for destination host
- **dst_host_srv_rerror_rate**: REJ error rate for destination service

### 5) Labels and Metadata
- **label** (last column): Contains attack types or "normal"
  - ✅ **normal**: Legitimate, harmless traffic
  - ❌ **Attack types**: neptune, smurf, apache2, satan, ipsweep, etc.
- **difficulty** (optional): Classification difficulty (can be dropped)

## Data Processing:

The training script will automatically convert the labels to binary classification:
- `normal` → **Harmless** (0)
- Any attack type → **Threat** (1)

## Value Ranges:
- **Binary fields**: 0 or 1
- **Rate fields**: 0.0 to 1.0  
- **Count fields**: Non-negative integers
- **Byte fields**: Non-negative integers
- **Categorical fields**: Must be encoded (protocol_type, service, flag)

## Preprocessing Steps:
1. Load CSV with all features
2. Encode categorical features (protocol_type, service, flag)
3. Normalize numeric features
4. Drop difficulty column if present
5. Convert labels to binary (normal vs attack)
6. Split into training and validation sets
