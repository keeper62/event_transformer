from models.token import LogTemplateMiner

if __name__ == "__main__":
    path = "data/xc40.log"
    
    tokenizer = LogTemplateMiner("drain3_state_xc40_new.bin")
    
    with open(path, 'r', errors='replace') as f:
        for line in f:
            log = line
            tokenizer.add_log_message(log)

    tokenizer.save_state()
        
    result = 1
    
# nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server [req-38101a0b-2096-447d-96ea-a692162415ae 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2477829