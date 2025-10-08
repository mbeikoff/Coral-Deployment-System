import Jetson.GPIO as GPIO
print(GPIO.JETSON_INFO)  # Should show {'TYPE': 'JETSON_ORIN_NANO', ...}
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)  # Pin 11 example
GPIO.output(11, GPIO.HIGH)
GPIO.cleanup()