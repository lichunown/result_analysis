# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:56:56 2020

@author: lichunyang23
"""
import os
import abc
import numpy as np
import warnings

from functools import reduce

class Result(np.ndarray):
    def __init__(self, *args, **kwargs):
        np.ndarray.__init__(*args, **kwargs)
    
    @abc.abstractmethod
    @property
    def value(self):
        pass
    
    @abc.abstractmethod
    @property
    def step(self):
        pass


def set_result_info(step_dim, value_dim):
    class TemplateResult(Result):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        @property
        def value(self):
            return self[:, value_dim]
        
        @property
        def step(self):
            return self[:, step_dim]
    return TemplateResult
    

class ResultSet(object):
    def __init__(self, set_dir=None, split_list:list=None, split_key='_', match_warn='warn'):
        """
        if input is None, do nothing.
        if set_dir is not none, will call `load_set` to load results.
            match_warn can be: 
                `ignore`: name in set_dir can not match split_list will do nothing.
                `warn`: name in set_dir can not match split_list will print warning.
                `raise`: name in set_dir can not match split_list will raise error.
        """
        self.split_list = split_list
        self.split_key = split_key
        self.set_dir = set_dir
        self.data = {}
        self.file_suffix = None
        
        if self.set_dir is not None:
            if split_list is None:
                raise ValueError('split_list must be a list')
            self._load_sets(set_dir, split_list, split_key, match_warn)
    
    @abc.abstractmethod
    def load_set(self, path):
        pass
    
    @staticmethod
    def without_suffix(name):
        name_without_suffix = '.'.join(name.split('.')[:-1])
        return name_without_suffix

    def split_name(self, name_without_suffix):
        name_split = name_without_suffix.split(self.split_key)
        return name_split
    
    def _load_sets(self, set_dir, split_list:list, split_key, match_warn):
        def match_warning(name):
            print(f'can not match {name}.')
        def match_ignore(name):
            pass
        def match_raise_error(name):
            raise ValueError(f'can not match {name}.')
        match_error_func = {
            'ignore': match_ignore,
            'warn': match_warning,
            'raise': match_raise_error,
        }
        result_set = {}
        for name in os.listdir(set_dir):
            save_name = self.without_suffix(name)
            name_split = self.split_name(save_name)
            if len(name_split) == len(split_list):
                result_path = os.path.join(set_dir, name)
                result_set[save_name] = self.load_set(result_path)
            else:
                match_error_func[match_warn](name)
        self.data = result_set
    
    def items(self):
        return self.data.items()
    
    def keys(self):
        return self.data.keys()

    def name_type(self, name):
        return dict(zip(self.split_list, self.split_name(name)))
    
    # set op
    def get(self, **kwargs):
        result_name = list(self.keys())
        for name in self.data:
            name_type = self.name_type(name)
            for key_name in kwargs:
                if kwargs[key_name] != name_type[key_name]:
                    result_name.remove(name)
                    break
        new_resultset = ResultSet()
        new_resultset.split_list = self.split_list
        new_resultset.split_key = self.split_key
        for name in result_name:
            new_resultset.data[name] = self.data[name]
        return new_resultset
    
    def copy(self):
        new_resultset = ResultSet()
        new_resultset.data = self.data.copy()
        new_resultset.split_list = self.split_list
        new_resultset.split_key = self.split_key
        new_resultset.set_dir = self.set_dir
        return new_resultset
    
    def meta(self, tag_k, tag_v):
        new_resultset = ResultSet()
        new_resultset.split_list = [tag_k] + self.split_list
        new_resultset.split_key = self.split_key
        
        for name, value in self.data.items():
            name_type = self.name_type(name)
            name_type[tag_k] = tag_v
            new_name = self.split_key.join([name_type.get(k) for k in new_resultset.split_list])
            new_resultset.data[new_name] = value
            
        return new_resultset
    
    def __add__(self, result_set):
        assert isinstance(result_set, ResultSet)
        
        if self.split_key != result_set.split_key:
            warnings.warn(f'c = a(split_key=`{self.split_key}`) adding b(split_key=`{result_set.split_key}`)' +
                          f', which will use `{self.split_key}` as c\'s split_key.')
        
        if tuple(self.split_list) != tuple(result_set.split_list):
            new_split_list = self.split_list + list((set(self.split_list) | set(result_set.split_list)) - set(self.split_list))
            warnings.warn(f'c = a(split_list=`{self.split_list}`) adding b(split_key=`{result_set.split_list}`)' +
                          f', which will use `{new_split_list}` as c\'s split_key.')
        else:
            new_split_list = self.split_list
            
        new_resultset = ResultSet()
        new_resultset.split_key = self.split_key
        new_resultset.split_list = new_split_list
        
        for name, value in result_set.data.items():
            name_type = result_set.name_type(name)
            new_name = self.split_key.join([str(name_type.get(k)) for k in new_resultset.split_list])
            new_resultset.data[new_name] = value
            
        for name, value in self.data.items():
            name_type = self.name_type(name)
            new_name = self.split_key.join([str(name_type.get(k)) for k in new_resultset.split_list])
            new_resultset.data[new_name] = value
        
        return new_resultset
    
    # calculating
    @property
    def shapes(self):
        return [item.shape for item in self.data.values()]
    
    @property
    def min_shape(self):
        np.min(self.shapes, 0)
    
    @property
    def is_same_shape(self):
        return all(map(lambda x:x==self.shapes[0], self.shapes))
    
    def _calculate_check(self):
        '''
        checking data have same dims.
        '''
        if not self.is_same_shape:
            raise ValueError('data must have same dim.')
            
    def mean(self):
        self._calculate_check()
        y = np.array(list(map(lambda x: x.value, self.data.values()))).mean(0)
        return y
    
    def std(self):
        self._calculate_check()
        y = np.array(list(map(lambda x: x.value, self.data.values()))).std(0)
        return y
    
    def var(self):
        self._calculate_check()
        y = np.array(list(map(lambda x: x.value, self.data.values()))).var(0)
        return y
    
    def step(self):
        self._calculate_check()
        step = np.array(list(map(lambda x: x.step, self.data.values())))
        if not np.all(map(lambda x:x==step[0], step)):
            raise ValueError('step not equal.')
        return step[0]
    
    def aligned_x(self):
        steps = np.array(list(map(lambda x: x.step, self.data.values())))
        aligned = list(reduce(lambda x, y: x & y, [set(step) for step in steps]))
        aligned.sort()
        return np.array(aligned)
    
    def aligned(self):
        x = self.aligned_x()
        # TODO
        new_resultset = ResultSet()
        new_resultset.split_key = self.split_key
        new_resultset.split_list = self.split_list
        
        for name, result in self.data.items():
            i = 0
            aligned_value = []
            for line in result:
                if i < len(x) and line.reshape(1, -1).step == x[i]:
                    aligned_value.append(line)
                    i += 1
            aligned_value = np.array(aligned_value).view(result.__class__)
            
            new_resultset.data[name] = aligned_value
        return new_resultset

    # other informations
    def __getitem__(self, n):
        return list(self.data.items())[n]
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self, **kwargs):
        '''
        env=None, lambda_type=None, lambda_=None, dtype=None, tag=None, seed=None
        '''
        return f'ResultSet(dir={self.set_dir}, len={len(self)})'
    
    def __repr__(self):
        return f'ResultSet in {self.set_dir} with {len(self)} results:\n' + str(list(self.data.keys())).replace('\', ', '\',\n')
        
    
    
    
#class BRACResultSet(ResultSet):
#    def load_set(self, path):
#        f = open(path, 'r')
#        if f.readline() == 'Wall time,Step,Value\n':
#            data = []
#            for line in f:
#                if line == '\n':
#                    break
#                wall_time, step, value = [float(i) for i in line.strip().split(',')]
#                data.append((wall_time, step, value))
#            return np.array(data).view(set_result_info(1, 2))
#        
        
#test = BRACResultSet('./test_example', 
#                 ['env', 'lambda_type', 'lambda', 'type', 'tag', 'seed'], split_key='+')   