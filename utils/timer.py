from time import time


class TimerError(Exception):
    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


# TODO 코드 깔끔정리 필요
class Timer:
    def __init__(self, print_msg, is_batch, model_type, det_task="NO", data_len=None, is_print_full_time=True, task_type=None):
        self.print_msg = print_msg
        # self.infer_len = infer_len
        self.is_batch = is_batch
        self.model_type = model_type
        self.det_task = det_task
        self.data_len = data_len
        self.is_print_full_time = is_print_full_time
        self.task_type = task_type
        self.time_results = []
        self._is_running = False


    def is_running(self):
        return self._is_running


    def __enter__(self):
        # self.start()

        if self.det_task == 'NO':
            task = self.model_type
        else:
            task = self.det_task
        print(f'{task.upper()} Time Checking... ', end="")

        return self


    def __exit__(self, type, value, traceback):
        print('Finish!')
        
        if self.is_print_full_time:
            final_time = self._cal_full_time()
            print(f'--> {self.print_msg}: {final_time:.3f}')
        
        # if self.is_batch:
        #     print(f'--> Batch mean time: {self._cal_mean_time(final_time)} (data length: {self.data_len})')
            
        print()
        self._is_running = False
        

    def start(self):
        # print(self.start.__name__)
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()


    def finish(self):
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        runtime = self._t_last - self._t_start
        self.set_time(runtime)


    def set_time(self, runtime):
        self.time = runtime
    

    def get_time(self):
        return self.time

   
    def _cal_full_time(self):
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start


    def _cal_mean_time(self, final_time):
        if not self._is_running:
            raise TimerError('timer is not running')
        return round((final_time/self.data_len), 3)


    def _cal_mid_time(self):
        if not self._is_running:
            raise TimerError('timer is not running')

        self._t_mid_start = time()
        mid_time = time() - self._t_mid_start
        self.time_results.append(mid_time)



def cal_total_mean_time(time_dict):
    total_time = 0.0
    mean_time = 0.0
    max_time = 0.0
    min_time = 0.0

    return_dict = dict()

    for task, results in time_dict.items():
        data_len = len(results)
        tmp_dict = dict()
        
        total_time = round(sum(results), 3)
        mean_time = round(total_time/data_len, 3)
        max_time = round(max(results), 3)
        min_time = round(min(results), 7)

        tmp_dict['total_time'] = total_time
        tmp_dict['mea_time'] = mean_time
        tmp_dict['max_time'] = max_time
        tmp_dict['min_time'] = min_time

        return_dict[task] = tmp_dict

    
    return return_dict


# Not Used
def get_det_pred_time(dict_pred):
    total_pred_time = None
    mean_pred_time = None
    max_infer = None
    min_infer = None
    all_time = []

    dict_all_pred_time = dict()

    for task, pred_time_per_task in dict_pred.items():
        all_time = []
        tmp_dict = dict()

        for _, pred_time in pred_time_per_task:
            all_time.append(pred_time)
        
        data_len = len(all_time)
        total_pred_time = sum(all_time)
        mean_pred_time = round(total_pred_time/data_len, 3)
        max_infer = max(all_time)
        min_infer = min(all_time)
        
        tmp_dict['total_time'] = round(total_pred_time, 3)
        tmp_dict['max_time'] = round(max_infer, 3)
        tmp_dict['min_time'] = round(min_infer, 3)
        tmp_dict['mean_time'] = round(mean_pred_time, 3)
        tmp_dict['data_length'] = data_len

        dict_all_pred_time[task.upper()] = tmp_dict
        # max_infer = max(pred_time_per_task)
        # min_infer = min(pred_time_per_task)
        # total_pred_time = sum(pred_time_per_task)
        # dict_total_pred_time[f'total_{task}'] = round(total_pred_time, 3)
        # dict_total_pred_time[f'max_{task}'] = round(max_infer, 3)
        # dict_total_pred_time[f'min_{task}'] = round(min_infer, 3)

        # if len(pred_time_per_task) > 1:
        #     mean_pred_time = round(total_pred_time/len(pred_time_per_task), 3)


    return dict_all_pred_time

# Not Used
def get_pred_time_manually(dict_time_results, pose_model_type):
    total_pred_time = None
    mean_pred_time = None
    max_infer = None
    min_infer = None
    all_time = []

    dict_all_pred_time = dict()
    # dict_mean_pred_time = dict()

    if pose_model_type == 'TopDown':
        for task, pred_time_per_task in dict_time_results.items():
            all_time = []
            tmp_dict = dict()

            for _, pred_time in pred_time_per_task:
                all_time.append(pred_time)
            
            data_len = len(all_time)
            total_pred_time = sum(all_time)
            mean_pred_time = round(total_pred_time/data_len, 3)
            max_infer = max(all_time)
            min_infer = min(all_time)
            
            tmp_dict['total_time'] = round(total_pred_time, 3)
            tmp_dict['max_time'] = round(max_infer, 3)
            tmp_dict['min_time'] = round(min_infer, 3)
            tmp_dict['mean_time'] = round(mean_pred_time, 3)
            tmp_dict['data_length'] = data_len

            dict_all_pred_time[task.upper()] = tmp_dict
            # max_infer = max(pred_time_per_task)
            # min_infer = min(pred_time_per_task)
            # total_pred_time = sum(pred_time_per_task)
            # dict_total_pred_time[f'total_{task}'] = round(total_pred_time, 3)
            # dict_total_pred_time[f'max_{task}'] = round(max_infer, 3)
            # dict_total_pred_time[f'min_{task}'] = round(min_infer, 3)

            # if len(pred_time_per_task) > 1:
            #     mean_pred_time = round(total_pred_time/len(pred_time_per_task), 3)


    elif pose_model_type == 'BottomUp':
        list_time_reuslts = list(dict_time_results.values())[0]
        model = list(dict_time_results.keys())[0]

        for _, time in list_time_reuslts:
            all_time.append(time)

        data_len = len(all_time)
        total_pred_time = sum(all_time)
        mean_pred_time = round(total_pred_time/data_len, 3)
        max_infer = max(all_time)
        min_infer = min(all_time)


        dict_all_pred_time['model'] = model.upper()
        dict_all_pred_time['total_time'] = round(total_pred_time, 3)
        dict_all_pred_time['max_time'] = round(max_infer, 3)
        dict_all_pred_time['min_time'] = round(min_infer, 3)
        dict_all_pred_time['mean_time'] = round(mean_pred_time, 3)


    # return dict_total_pred_time, dict_mean_pred_time

    return dict_all_pred_time


# def infer_with_sync():
