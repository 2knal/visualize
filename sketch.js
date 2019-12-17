// GLOBALS
// Regression

let X = [];
let y = [];
let curveX = [];
let losses = [];
let drawCurve = false;
let dragging = false;
let a, b, c, d;
let learningRate = 0.25;
let opti = 'adam';
let optimizer = tf.train.adam(learningRate);

// Initializing slider value
$('#slider_value').html($('#slider').val());

$(document).on('input', '#slider', function() {
    $('#slider_value').html( $(this).val() );
});
// Hiding reset and vis button initially
$('#reset').hide();
$('#vis').hide();

function handleVisualisation() {
    tfvis.visor().toggle();
    setInterval(() => watchTraining(), 1000);
}

function watchTraining() {
    const series = losses.map((y, x) => ({ x, y, }));

    const data = { values: [series] }
    // console.log(data)
    // Render to visor
    const surface = { name: 'Graphs, yayy! (LOSS)', tab: 'Training' };
    tfvis.render.linechart(surface, data, { zoomToFit: true });
}

function handleReset() {
    drawCurve = false;
    X = [];
    y = [];
    curveX = [];
    optimizer = null;
    losses = []
    // console.log('clicked')
    $('#start').show();
    $('#reset').hide();
    $('#vis').hide();
    $('#drop_me').show();
}

function handleStart() {
    drawCurve = true;
    optimizer = opti === 'sgd'? tf.train.sgd(learningRate): tf.train.adam(learningRate);
    // Assigning random values to weights
    a = tf.variable(tf.scalar(random(0, 1)));
    b = tf.variable(tf.scalar(random(0, 1)));
    c = tf.variable(tf.scalar(random(0, 1)));
    d = tf.variable(tf.scalar(random(0, 1)));
    // console.log('clicked')
    $('#reset').show();
    $('#vis').show();
    $('#start').hide();
    $('#drop_me').hide();
}

// Slider to modify learning rate
$(document).on('input', '#slider', function() {
    learningRate = parseFloat($(this).val());
    optimizer = opti === 'sgd'? tf.train.sgd(learningRate): tf.train.adam(learningRate);
    // console.log(opti);
});

// Optimizer selection
$('#dropdown').click(function() {
    opti = $('#dropdown option:selected').val();
    optimizer = opti === 'sgd'? tf.train.sgd(learningRate): tf.train.adam(learningRate);
    // console.log(opti, optimizer);
});

// Degree of polynomial
let degree = 1;
$(document).on("input", function() {
    degree = parseInt($('input[type=radio]:checked').val());
    // console.log(degree)
});

function setup() {
    let cnv = createCanvas(600, 600);
    cnv.id('canvas')
    cnv.parent('main')

    // Assigning random values to weights
    a = tf.variable(tf.scalar(random(0, 1)));
    b = tf.variable(tf.scalar(random(0, 1)));
    c = tf.variable(tf.scalar(random(0, 1)));
    d = tf.variable(tf.scalar(random(0, 1)));
}

function mousePressed() {
    dragging = true;
}

function mouseReleased() {
    dragging = false;
}

function predict(X_vals) {
    let temp = tf.tensor1d(X_vals);

    if (degree === 1) {
        // ax + b
        return a.mul(temp).add(b);
    } else if (degree === 2) {
        // ax^2 + bx^1 + c
        return a.mul(temp.square()).add(b.mul(temp)).add(c);
    }
    // ax^3 + bx^2 + cx + d
    return a.mul(tf.pow(temp, 3)).add(b.mul(temp.square())).add(c.mul(temp)).add(d);
}

function loss(X_vals, labels) {
    return X_vals.sub(labels).square().mean();
}

function train() {
    let metric = {
        history: {
            loss:[]
        }
    };
    metric.history.loss.push(optimizer.minimize(() => loss(predict(X), y), returnCost=true, [a, b, c, d]).dataSync()[0]);
    return metric;
}

function draw() {
    background('#444');
    stroke(255);
    strokeWeight(8);
    
    // Drawing points
    for(let i=0; i<X.length; i++) {
        const px = map(X[i], 0, 1, 0, width);
        const py = map(y[i], 0, 1, height, 0);
        point(px, py);
    }

    if (dragging) {
        if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
            const px = map(mouseX, 0, width, 0, 1);
            const py = map(mouseY, 0, height, 1, 0);
        
            X.push(px);
            y.push(py);
        }
    } else {
        // Memory management
        tf.tidy(() => {
            if (X.length > 0 && drawCurve) {
                // Training...
                cost = train();
                // console.log(cost.dataSync());
                losses.push(cost.history.loss[0]);
            }
        });
    }

    if (drawCurve) {
        // Drawing a curve | line
        curveX = [];
        for(let i=0; i<=1; i+=0.005) {
            curveX.push(i)
        }
        
        let curveY = tf.tidy(() => predict(curveX));
        curveY = curveY.dataSync();

        beginShape();
        noFill();
        stroke(255);
        strokeWeight(2);
        for (let i=0; i<curveY.length; i++) {
            const x = map(curveX[i], 0, 1, 0, width);
            const y = map(curveY[i], 0, 1, height, 0);
            vertex(x, y);
        }
        endShape();
    }
}