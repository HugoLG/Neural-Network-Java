import java.util.Vector;
import java.io.*;

/**
 * FeedForward with Backpropagation Artificial Neural Network
 * This class can be use to create a FeedForward with Backpropagation
 * neural network, the parameters values of the network are customizable,
 * also the number of neurons and the number of hidden layers are defined
 * by the user when creating the neural network object.
 *
 * <b> Note: </b> This class needs the Neuron class.
 *
 * @author Hugo León Garza
 * @version 1.0
 * @since July-2016
 *
 */

public class NeuralNetwork {

    private double lambda;
    private double eta;
    private double momentum;
    private Vector<Vector<Neuron>> neuralNetwork;
    private int numInputNeurons;
    private int numOutputNeurons;
    private int numHiddenNeurons;

    //Constructor
    public NeuralNetwork(Double l, Double e, Double a, int inNeurons, int numHiddenLayers, int hiddenNeurons, int outNeurons, boolean addHiddenBias) {
        this.lambda = l;
        this.eta = e;
        this.momentum = a;
        this.numInputNeurons = inNeurons;
        this.numOutputNeurons = outNeurons;
        this.numHiddenNeurons = hiddenNeurons;
        this.neuralNetwork = new Vector<Vector<Neuron>>();

        Vector<Neuron> inputLayer = new Vector<Neuron>();
        for (int i=0; i<inNeurons; i++) {
            Neuron n = new Neuron(numHiddenNeurons);
            inputLayer.add(n);
        }
        neuralNetwork.add(inputLayer);

        Vector<Neuron> hiddenLayer = new Vector<Neuron>();
        if(addHiddenBias) {
            hiddenLayer.add(new Neuron(true));
        }
        for(int i=0; i<numHiddenLayers; i++) {
            for(int j=0; j<hiddenNeurons; j++) {
                Neuron n;
                if((j+1) < hiddenNeurons) {
                    n = new Neuron(numHiddenNeurons);
                } else {
                    n = new Neuron(numOutputNeurons);
                }
                hiddenLayer.add(n);
            }
            neuralNetwork.add(hiddenLayer);
        }

        Vector<Neuron> outputLayer = new Vector<Neuron>();
        for (int i=0; i<outNeurons; i++) {
            Neuron n = new Neuron();
            outputLayer.add(n);
        }
        neuralNetwork.add(outputLayer);
    }

    //METHODS
    public void printNetwork() {
        System.out.println("Lambda: " + this.lambda);
        System.out.println("Eta: " + this.eta);
        System.out.println("Momentum: " + this.momentum);
        System.out.println("Neural Network... ");
        System.out.println("Input Layer: ");
        for(int i=0; i<neuralNetwork.get(0).size(); i++) {
            neuralNetwork.get(0).get(i).print();
            System.out.print(" ");
        }
        System.out.println();
        System.out.println("Hidden Layers: ");
        for(int i=1; i<neuralNetwork.size()-1; i++) {
            for(int j=0; j<neuralNetwork.get(i).size(); j++) {
                neuralNetwork.get(i).get(j).print();
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("Output Layer: ");
        for(int i=0; i<neuralNetwork.get(neuralNetwork.size()-1).size(); i++) {
            neuralNetwork.get(neuralNetwork.size()-1).get(i).print();
            System.out.print(" ");
        }
        System.out.println();
    }

    public void trainNetwork(String trainingFilename) {

        File file = new File(trainingFilename);

        for(int epoch=0; epoch<epochLimit; epoch++) {

            //read file and pass each row through training process
            Scanner inputFile = new Scanner(file);
            while(inputFile.hasNext()) {
                String line = inputFile.nextLine();
                String data[] = line.split(",");
                Vector<Double> inputs = new Vector<Double>();

                for(int i=0; i<data.length; i++) {
                    inputs.add(Double.parseDouble(data[i]));
                    //System.out.println("Data input #" + i + ":  " + inputs.get(i));
                }

                feedForward(inputs);
                inputErrors = calculateErrors();
                backPropagation(inputErrors);
            }

            //run network through validation data and check validation error
        }

        //read training examples
        //send each example to forward and backpropagation
        //feedforward
        //calculate output
        //calculate error;
        //backpropagation
        //calculate local gradient
        //calculate change of weights
        //update weights
        //
        //calculate training epoch error;
        //validation forward
        //calculate validation error;
    }

    /**
     * This method is used to make the Neural Network go through the
     * Feed Forward process one time with the training sample that was
     * received as parameter.
     *
     * @param trainingSample This is a vector with the input values for each neuron. (Values must be a Double)
     */
    public void feedForward(Vector<Double> trainingSample) {
        for(int i=0; i<trainingSample.size(); i++) {
            neuralNetwork.get(0).get(i).setValue(trainingSample.get(i));
        }

        //All layers will use the same activation function, if a layer uses another activation function
        //modifications to this for has to be made
        for(int i=1; i<neuralNetwork.size(); i++) {

            for(int j=0; j<neuralNetwork.get(i).size(); j++) {
                double tmpV = 0;
                for(int k=0; k<neuralNetwork.get(i-1).size(); k++) {
                    tmpV += (neuralNetwork.get(i-1).get(k).getValue() * neuralNetwork.get(i-1).get(k).getWeight(j));
                }
                neuralNetwork.get(i).get(j).setValue(tmpV);
                neuralNetwork.get(i).get(j).calculateH(1, this.lambda);
            }
        }
    }

    /**
     *
     *
     *
     *
     */
    public Vector<Double> predictValue(Vector<Double> inputs) {
        for(int i=0; i<inputs.size(); i++) {
            this.neuralNetwork.get(0).get(i).setValue(inputs.get(i));
        }

        //All layers will use the same activation function, if a layer uses another activation function
        //modifications to this for has to be made
        for(int i=1; i<this.neuralNetwork.size(); i++) {

            for(int j=0; j<this.neuralNetwork.get(i).size(); j++) {
                double tmpV = 0;
                for(int k=0; k<this.neuralNetwork.get(i-1).size(); k++) {
                    tmpV += (this.neuralNetwork.get(i-1).get(k).getValue() * this.neuralNetwork.get(i-1).get(k).getWeight(j));
                }
                this.neuralNetwork.get(i).get(j).setValue(tmpV);
                this.neuralNetwork.get(i).get(j).calculateH(1, this.lambda);
            }
        }

        Vector<Double> predictedOutput = new Vector<Double>();
        for(int i=0; i<this.neuralNetwork.get(this.neuralNetwork.size()-1).size(); i++) {
            predictedOutput.add(this.neuralNetwork.get(this.neuralNetwork.size()-1).get(i).getValue());
        }

        return predictedOutput;
    }

    /**
     *
     *
     */
    public Vector<Double> calculateErrors(Vector<Double> target) {

        Vector<Double> tmp = new Vector<Double>();

        for(int i=0; i<target.size(); i++) {
            tmp.add(target.get(i) - neuralNetwork.get(neuralNetwork.size()-1).get(i).getValue());
        }

        return tmp;

    }

    /**
     *
     *
     *
     */
    public Vector<Double> getOutputValues() {

        Vector<Double> tmp = new Vector<Double>();

        for(int i=0; i<neuralNetwork.get(neuralNetwork.size()-1).size(); i++) {
            tmp.add(neuralNetwork.get(neuralNetwork.size()-1).get(i).getH());
        }

        return tmp;
    }

    /**
     *
     *
     *
     */
    public void backPropagation(Vector<Double> errors) {

        //First do the back propagation in the last layer
        Vector<Double> localGradientsOutputLayer = new Vector<Double>();
        for(int i=0; i<neuralNetwork.get(neuralNetwork.size()-1).size(); i++) {
            Double opt = neuralNetwork.get(neuralNetwork.size()-1).get(i).getH();
            localGradientsOutputLayer.add(this.lambda*opt*(1.0-opt)*errors.get(i));
        }

        for(int i=0; i<neuralNetwork.get(neuralNetwork.size()-2).size(); i++) {
            Vector<Double> weights = neuralNetwork.get(neuralNetwork.size()-2).get(i).getWeights();
            Vector<Double> prevDeltaWeights = neuralNetwork.get(neuralNetwork.size()-2).get(i).getPreviousDeltaWeights();
            Vector<Double> newWeights = new Vector<Double>();
            Vector<Double> newPrevDeltaWeights = new Vector<Double>();

            for( int j=0; j<neuralNetwork.get(neuralNetwork.size()-1).size(); j++) {
                Double deltaWeight = this.eta*neuralNetwork.get(neuralNetwork.size()-2).get(i).getH()*localGradientsOutputLayer.get(j)
                    + this.momentum*prevDeltaWeights.get(j);
                newWeights.add(weights.get(j)+deltaWeight);
                newPrevDeltaWeights.add(deltaWeight);
            }

            neuralNetwork.get(neuralNetwork.size()-2).get(i).setWeights(newWeights);
            neuralNetwork.get(neuralNetwork.size()-2).get(i).setPreviousDeltaWeights(newPrevDeltaWeights);

        }

        //Back propagation learning in the rest of the layers
        Vector<Double> prevGradients = localGradientsOutputLayer;
        for(int i=neuralNetwork.size()-2; i>0; i--) {

            Vector<Double> localGradients = new Vector<Double>();
            for(int j=0; j<neuralNetwork.get(i).size(); j++) {
                Double sum = 0.0;
                Vector<Double> neuronWeights = neuralNetwork.get(i).get(j).getWeights();
                for(int k=0; k<neuronWeights.size(); k++) {
                    sum += prevGradients.get(k)*neuronWeights.get(k);
                }
                Double opt = neuralNetwork.get(i).get(j).getH();
                localGradients.add(this.lambda*opt*(1.0-opt)*sum);
            }

            for(int j=0; j<neuralNetwork.get(i-1).size(); j++) {
                Vector<Double> weights = neuralNetwork.get(i-1).get(j).getWeights();
                Vector<Double> prevDeltaWeights = neuralNetwork.get(i-1).get(j).getPreviousDeltaWeights();
                Vector<Double> newWeights = new Vector<Double>();
                Vector<Double> newPrevDeltaWeights = new Vector<Double>();

                for(int k=0; k<neuralNetwork.get(i).size(); k++) {
                    Double deltaWeight = this.eta*localGradients.get(k)*neuralNetwork.get(i-1).get(k).getH()
                        + this.momentum*prevDeltaWeights.get(k);
                    newPrevDeltaWeights.add(deltaWeight);
                    newWeights.add(weights.get(j)+deltaWeight);
                }

                neuralNetwork.get(i-1).get(j).setWeights(newWeights);
                neuralNetwork.get(i-1).get(j).setPreviousDeltaWeights(newPrevDeltaWeights);

            }

            prevGradients = localGradients;

        }


    }

    //SETTERS
    public void setLambda(double l) {
        this.lambda = l;
    }

    public void setMomentum(double a) {
        this.momentum = a;
    }

    public void setEta(double e) {
        this.eta = e;
    }

    public void setNeuralNetwork(Vector<Vector<Neuron>> nn) {
        this.neuralNetwork = nn;
    }

    public void setNumInputNeurons(int i) {
        this.numInputNeurons = i;
    }

    public void setNumOutputNeurons(int i) {
        this.numOutputNeurons = i;
    }

    public void setNumHiddenNeurons(int i) {
        this.numHiddenNeurons = i;
    }

    //GETTERS
    public double getLambda() {
        return this.lambda;
    }

    public double getMomentum() {
        return this.momentum;
    }

    public double getEta() {
        return this.eta;
    }

    public Vector<Vector<Neuron>> getNeuralNetwork() {
        return this.neuralNetwork;
    }

    public int getNumInputNeurons() {
        return this.numInputNeurons;
    }

    public int getNumOutputNeurons() {
        return this.numOutputNeurons;
    }

    public int getNumHiddenNeurons() {
        return this.numHiddenNeurons;
    }
}
