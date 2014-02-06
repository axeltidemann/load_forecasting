import Oger, mdp, pdb
import numpy as NP

class StaticNode(Oger.nodes.ReservoirNode):
    """ Extends the Reservoir node for static classification by letting the inner dynamics of the reservoir settle before the final state of that timestep is
stored.

Note: in the original paper, the transfer function is not used. It is not clear why one shouldn't use the tanh function,
however, this is not hardcoded. Use the identity function as an input parameter if this behaviour is desired. It seems to me that
the network performs better when the tanh transfer function is used. 

Author: Axel Tidemann
"""
    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        steps = x.shape[0]
        
        # Pre-allocate the state vector, adding the initial state. All zeros.
        states = mdp.numx.zeros((steps, self.output_dim))

        # A vector to store how many steps were needed to stabilize the reservoir.
        stabilize = mdp.numx.zeros(steps)

        # Loop over the input data and compute the reservoir states.
        for n in range(steps):
            # Let the reservoir stabilize before collection.
            previous_state = states[n,:]
            current_state = self.nonlin_func(mdp.numx.dot(self.w, states[n, :]) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)

            i = 0
            # We continue until a 0.1% change. Formula taken from Wikipedia for % difference (not percent error).
            while abs(NP.sum(previous_state - current_state))/max(abs(NP.sum(previous_state)), abs(NP.sum(current_state))) > 0.001: 
                previous_state = current_state
                # Added flattening of previous_state in the following line, 2012-07-18. Somehow the transposing did not happen before, or
                # maybe MDP was more tolerant.
                current_state = self.nonlin_func(mdp.numx.dot(self.w, NP.ndarray.flatten(previous_state)) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)
                i += 1

            stabilize[n] = i
            states[n, :] = current_state
            self._post_update_hook(states, x, n)    

        print 'StaticNode: Steps to stabilize the reservoir (avg std min max)', NP.average(stabilize), NP.std(stabilize), min(stabilize), max(stabilize)

        #print NP.max(states), NP.min(states), NP.average(states)
        
        return states


##### Testing #####
if __name__ == "__main__":

    #Generate random vectors
    NP.random.seed()
    x = NP.random.randn(100,20) # (number of cases, number of features)
    #Generate target vector - one vector for each case.
    y = NP.eye(100)

    #Create ESN
    reservoir = StaticNode(input_dim = x.shape[1], output_dim = 20, spectral_radius = 0.55) #Too large reservoir -> trouble.
    readout = Oger.nodes.RidgeRegressionNode()

    flow = mdp.hinet.FlowNode(reservoir + readout)
    flow.train(x, y)
    flow.stop_training()

    ytest = flow(x) 

    # See how well the classification works, e.g. if the highest activated output node is the correct one. 
    c = 0
    for i in range(y.shape[0]):
        if NP.argmax(ytest[i,:]) == NP.argmax(y[i,:]):
            c += 1

    print 'Absolute error:', NP.mean(ytest - y), 'Classfication rate:', 100*c/y.shape[0], '%'

