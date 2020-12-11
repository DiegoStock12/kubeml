package scheduler

import (
	"container/list"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"sync"
)


// queue is the internal type used to queue
// the elements in the scheduler
type queue = list.List

// SchedulerQueue is the queue that will be used for the scheduler
type SchedulerQueue struct {

	// lock used to access the tasks in the queue
	lock sync.RWMutex

	// trainQ holds the tasks that are running and have priority over the
	// tasks that are submitted
	trainQ *queue

	// waitQ holds the tasks that are submitted but are still waiting
	// cause there are not enough resources for them TODO we need a way to check this
	waitQ *queue
}

// pushTrainTask pushes the task so it can be analyzed
// and given a new parallelism level
func (sq *SchedulerQueue) pushTask(task *api.TrainTask)  {
	sq.lock.Lock()
	defer sq.lock.Lock()

	// Insert a new TrainTask in the queue
	sq.trainQ.PushBack(task)

}

// popTask returns the next element from the training queue
func (sq *SchedulerQueue) popTask() (*api.TrainTask, error)  {
	sq.lock.Lock()
	defer sq.lock.Unlock()

	if sq.trainQ.Len() == 0 {
		return nil, fmt.Errorf("queue is empty")
	}

	// get the first element and remove it from the
	// linked list
	e := sq.trainQ.Front()
	sq.trainQ.Remove(e)

	// Return the value as a train task
	return e.Value.(*api.TrainTask), nil
}

// TODO how will the queues interact?
// pushRequest pushes requests into the waiting request
func (sq *SchedulerQueue) pushRequest(req *api.TrainRequest) {
	sq.lock.Lock()
	defer sq.lock.Unlock()

	//sq.waitQ.PushBack(req)

	// right now just create a task and push it to queue
	t := api.TrainTask{
		Parameters:  *req,
		Parallelism: -1,
		JobId:       createJobId(),
		ElapsedTime: -1,
	}
	sq.trainQ.PushBack(t)

}


