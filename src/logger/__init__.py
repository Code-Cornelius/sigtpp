import logging

# Step 1: Define custom log level
FLAG_LEVEL = 80
logging.addLevelName(FLAG_LEVEL, "FLAG")


# Step 2: Add .flag method to the Logger class
def flag(self, message, *args, **kwargs):
    if self.isEnabledFor(FLAG_LEVEL):
        self._log(FLAG_LEVEL, message, args, **kwargs)


logging.Logger.flag = flag  # this is the key line
