async function run(model){
    const csvUrl = 'https://raw.githubusercontent.com/Yoshibansal/BrowserML-CSV/main/iris.csv';
    const trainingData = tf.data.csv(csvUrl, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });

    const numOfFeatures = (await trainingData.columnNames()).length - 1;

    const convertedData =
          trainingData.map(({xs, ys}) => {
              const labels = [
                    ys.species == "setosa" ? 1 : 0,
                    ys.species == "virginica" ? 1 : 0,
                    ys.species == "versicolor" ? 1 : 0
              ] 
              return{ xs: Object.values(xs), ys: Object.values(labels)};
          }).batch(10);
    
    await model.fitDataset(convertedData, 
                     {epochs:150,
                      callbacks:{
                          onEpochEnd: async(epoch, logs) =>{
                                        var temp = "Epoch: " + epoch
                                                            + "  Loss: " + logs.loss
                                        console.log(temp);
                                        document.getElementById("boxed").innerHTML = temp;
                          }
                      }});
}


const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [4], activation: "sigmoid", units: 5}))
model.add(tf.layers.dense({activation: "softmax", units: 3}));
    
model.compile({loss: "categoricalCrossentropy", optimizer: tf.train.adam(0.06)});
model.summary();

var button1 = document.getElementById("btn");
button1.disabled = true;

var options = document.getElementById("input-ex");
options.disabled = true;

run(model).then(() => {
    document.getElementById("status").innerText = "Model is ready to predict new values";
    button1.disabled = false;
    options.disabled = false;
});

function predict() {
    var string = (document.getElementById("input-ex").value);
    var array = JSON.parse(string);
    console.log(array);

    var ex1 = [4.4, 2.9, 1.4, 0.2];
    var ex2 = [5.8, 2.7, 5.1, 1.9];
    var ex3 = [6.4, 3.2, 4.5, 1.5];

    ///Original outputs: for comparison 
    if(JSON.stringify(ex1)==JSON.stringify(array)){
        document.getElementById("original").innerHTML = 'Setosa';
    }else if(JSON.stringify(ex2)==JSON.stringify(array)) {
        document.getElementById("original").innerHTML = 'Virginica';
    }else {
        document.getElementById("original").innerHTML = 'Versicolor';
    }

    //Prediction by model
    const testVal = tf.tensor2d(array, [1, 4]);
    
    const prediction = model.predict(testVal);
    const pIndex = tf.argMax(prediction, axis=1).dataSync();
    
    const classNames = ["Setosa", "Virginica", "Versicolor"];
    
    var ar = prediction.arraySync()[0];
    console.log(ar);

    document.getElementById("prediction").innerHTML = classNames[pIndex];

    var Original = JSON.stringify(document.getElementById("original").innerHTML);
    var Prediction = JSON.stringify(document.getElementById("prediction").innerHTML);
    document.getElementById("isCorrect").innerHTML = "Is prediction correct: " + (Original==Prediction);

    var data = [{
        type: 'bar',
        x: ar,
        y: classNames,
        orientation: 'h',
        marker: {
            color: 'rgb(253,106,2)',
            opacity: 0.5,
            line: {
                color: 'rgb(253,106,2)',
                width: 1.5
            }
        }
      }];

    var layout = {
        autosize: true,
        width: 700,
        height: 500,
        margin: {
          l: 70,
          r: 50,
          b: 100,
          t: 100,
          pad: 4
        }};
      
    Plotly.newPlot('myDiv', data, layout, {displayModeBar: false});
}
