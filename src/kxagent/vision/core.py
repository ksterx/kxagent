from ..core import Brain, Interpreter


class ImageProcessor(Brain):
    def __init__(self, model):
        super().__init__(model)


class ImageSensor(Interpreter):
    def __init__(self, name, data):
        super().__init__(name, data)

    def process(self):
        return SensorData(self.name, self.data)

    def read(self):
        return self.data

    def send(self):
        pass
