from enum import Enum

from .handlers import BirdeyeHandler, MessageHandler, CameraHandler, LidarHandler, CollisionHandler

class HandlerType(Enum):
    ''' User-defiend data sources '''
    RGB_CAMERA = 'camera'
    LIDAR = 'lidar'
    COLLISION = 'collision'
    BIRDEYE = 'birdeye'
    MESSAGE = 'message'
    SPECTATOR = 'spectator'

HANDLER_DICT = {
    HandlerType.BIRDEYE: BirdeyeHandler,
    HandlerType.MESSAGE: MessageHandler,
    HandlerType.RGB_CAMERA: CameraHandler,
    HandlerType.LIDAR: LidarHandler,
    HandlerType.COLLISION: CollisionHandler,
}
