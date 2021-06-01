import cog
from fastai.vision.all import *

class AnimalVisionModel(cog.Model):
    def setup(self):
        # Use FastAI instead of torch
        # A high-level interface on top of torch
        self.learner = load_learner("export.pkl")

    @cog.input("input", type=cog.Path, help="image of cat or dog (or other animal)")
    def predict(self, input):
        output = self.learner.predict(input)
        return str(output)