import torch
from dataclasses import dataclass

@dataclass
class InstanceGroup:
    def __init__(self, name, device='cpu'):
        self.name = name
        self.device = device
        self.insts = []
        self.name_to_id = {}
        self.id_to_name = {}
        self.ids = None
        
    def add(self, inst):
        self.insts.append(inst)

    @property
    def num(self) -> int:
        return len(self.insts)
        
    def __len__(self):
        return len(self.insts)
        
    def create_mappings(self, start_id=0):
        for idx, inst in enumerate(self.insts):
            name = inst.getName()
            current_id = start_id + idx
            self.name_to_id[name] = current_id
            self.id_to_name[current_id] = name
            
        self.ids = torch.arange(start_id, start_id + len(self.insts), device=self.device)
        return start_id + len(self.insts)

    def get_id(self, name):
        return self.name_to_id.get(name)
    
    def get_name(self, id):
        return self.id_to_name.get(id)
    
    def has_name(self, name):
        return name in self.name_to_id
    
    def has_id(self, id):
        return id < self.num