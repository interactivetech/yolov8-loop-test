
from determined.pytorch import DataLoader, LRScheduler, PyTorchTrial, PyTorchTrialContext

class YoloV8Trial(PyTorchTrial):
    
    def __init__(self,context: PyTorchTrialContext) -> None:
        '''
        '''
        self.context = context
        self.model = self.context.wrap_model(self.model)
        self.optimizer = self.context.wrap_optimizer()
        self.dataset = 
        self.scheduler = self.context.wrap_lr_scheduler(x,LRScheduler.StepMode.MANUAL_STEP)
        
        
    def build_training_data_loader(self) -> None:
        '''
        '''
    def build_validation_data_loader(self) -> None:
        '''
        '''
    def train_batch(self,batch,epoch_idx, batch_idx):
        '''
        '''
    def evaluate_batch(self,batch):
        '''
        '''