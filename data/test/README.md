# Test Data Directory

This directory is used for storing:
1. CSV files uploaded through the web interface
2. Prediction output files

## File Usage:

- **sample_input.csv** - CSV files uploaded via the web app are saved here temporarily
- Files with **_predictions.csv** suffix contain the original data plus prediction results

## CSV Format for Testing:

Your test CSV should have the same features (columns) as the training data, but does NOT need to include a label column since the model will predict it.

### Required Feature Categories:

#### 1) Basic Connection Features
- **duration**: Length of connection in seconds
- **protocol_type**: tcp, udp, or icmp
- **service**: http, ftp, telnet, smtp, etc.
- **flag**: Connection status (SF, S0, REJ, etc.)
- **src_bytes**: Bytes sent from source to destination  
- **dst_bytes**: Bytes sent from destination to source

#### 2) Content-Based Features
- **land**: 0 or 1 (1 if source and destination IP/port are same)
- **wrong_fragment**: Number of incorrect IP fragments
- **urgent**: Number of urgent packets
- **hot**: Number of sensitive operations
- **num_failed_logins**: Failed login attempts count
- **logged_in**: 0 or 1 (1 if login successful)
- **num_compromised**: Number of compromised conditions
- **root_shell**: 0 or 1 (1 if root shell obtained)
- **su_attempted**: 0 or 1 (1 if "su root" attempted)
- **num_root**: Number of root accesses
- **num_file_creations**: Files created during connection
- **num_shells**: Shell prompts opened
- **num_access_files**: Access count to sensitive files
- **num_outbound_cmds**: Always 0 in NSL-KDD (can be omitted)
- **is_host_login**: 0 or 1 (1 if login belongs to host list)
- **is_guest_login**: 0 or 1 (1 if guest login)

#### 3) Time-Based Traffic Features (calculated over last 2 seconds)
- **count**: Connections to same host
- **srv_count**: Connections to same service  
- **serror_rate**: SYN error percentage (0.0-1.0)
- **srv_serror_rate**: SYN error rate for same service
- **rerror_rate**: REJ error percentage (0.0-1.0)
- **srv_rerror_rate**: REJ error rate for same service
- **same_srv_rate**: Same service connection fraction
- **diff_srv_rate**: Different service connection fraction
- **srv_diff_host_rate**: Same service, different hosts rate

#### 4) Host-Based Traffic Features (calculated over last 100 connections)  
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

### Data Format Requirements:

- **CSV format** with comma separators
- **No header row** (or ensure column names match exactly)
- **Numeric values** for count, rate, and byte fields
- **String values** for categorical fields:
  - protocol_type: "tcp", "udp", "icmp"
  - service: "http", "ftp", "telnet", "smtp", etc.
  - flag: "SF", "S0", "REJ", etc.
- **Binary values** (0 or 1) for boolean fields
- **Rate values** between 0.0 and 1.0

### Example CSV Structure:
```
0,tcp,http,SF,239,486,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00
```

### Important Notes:

- **NO label column required** - the model will predict whether traffic is "Threat" or "Harmless"
- **Column order matters** - must match training data exactly
- **Missing values** should be filled with appropriate defaults (0 for counts, 0.0 for rates)
- **Categorical values** must use exact strings from training data

The web application will automatically handle the prediction and display results showing:
- Total records processed
- Number of threats detected  
- Number of harmless connections
- Detailed table with individual predictions
