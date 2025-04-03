from models.token import LogTokenizer

if __name__ == "__main__":
    path = "data/Linux.log"
    
    tokenizer = LogTokenizer("drain3_state_linux.bin")
    
    with open(path, "r") as f:
        for line in f:
            log = line.split(": ")[-1]
            tokenizer.add_log_message(log)

    tokenizer.save_state()
        
        