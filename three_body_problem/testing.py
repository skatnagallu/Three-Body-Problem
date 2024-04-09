from direct.showbase.ShowBase import ShowBase

class MyGame(ShowBase):
    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)
        star_model = loader.loadModel('../models/Sun.glb')
        star_model.reparentTo(render)

game = MyGame()
game.run()