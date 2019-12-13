// TODO Animate workers when performing tasks or swapping to show activity
// TODO Add memory usage dial around outside of workers
// TODO Add clients
// TODO Show future retrieval
// TODO Show graph submission
// TODO Handle window resize

class Dashboard {
    constructor() {
        this.workers = []
        this.scheduler = "scheduler"
        this.schedulerNode = null
        this.dashboard = document.getElementById('vis')
        this.transfers = []
        this.tasks = {}
    }

    handle_event(event) {
        switch (event['name']) {
            case 'pong':
                break;
            case 'worker_join':
                this.add_worker(event['id']);
                break;
            case 'remove_worker':
                this.remove_worker(event['id']);
                break;
            case 'start_task':
                this.start_task(event['id'], event['task_name']);
                break;
            case 'end_task':
                this.end_task(event['id'], event['task_name']);
                break;
            case 'start_transfer':
                // let start_worker = this.tasks[event['key']]['worker']
                this.start_transfer(event['start_worker'], event['end_worker']);
                break;
            case 'end_transfer':
                this.end_transfer(event['start_id'], event['end_id']);
                break;
            case 'start_swap':
                this.start_swap(event['id']);
                break;
            case 'end_swap':
                this.end_swap(event['id']);
                break;
            case 'killed_worker':
                this.kill_worker(event['id']);
                break;
            case 'reset':
                this.reset();
                break;
            case 'transition':
                if (event['action'] === "compute") {
                    this.start_task(event['worker_id'], event['key'], event['color']);
                    setTimeout(function () { this.end_task(event['worker_id'], event['key']) }.bind(this), event['stop'] - event['start']);
                    break;
                } else if (event['action'] === "transfer") {
                    let randomWorker = this.workers[Math.floor(Math.random() * this.workers.length)]; // Until we can work out where the tranfer is coming from let's illustrate with a random worker
                    let arc = this.start_transfer(randomWorker, event['worker_id']);
                    setTimeout(function () { this.end_transfer(randomWorker, event['worker_id'], arc) }.bind(this).bind(arc), Math.max(250, (event['stop'] - event['start']) * 1000));
                    break;
                }
            default:
                console.log("Unknown event " + event['name']);
                console.log(event);
        }
    }

    add_scheduler() {
        this.schedulerNode = document.createElementNS('http://www.w3.org/2000/svg', "circle");
        this.schedulerNode.setAttributeNS(null, "id", this.scheduler);
        this.schedulerNode.setAttributeNS(null, "class", "node scheduler");
        this.dashboard.appendChild(this.schedulerNode);
        gsap.fromTo("#" + this.scheduler, { r: 0, cx: "50%", cy: "50%" }, { r: 30, cx: "50%", cy: "50%", duration: 0.25 });
    }

    add_worker(id) {
        let workerNode = document.createElementNS('http://www.w3.org/2000/svg', "circle");
        workerNode.setAttributeNS(null, "id", id);
        workerNode.setAttributeNS(null, "r", "0");
        workerNode.setAttributeNS(null, "cx", "50%");
        workerNode.setAttributeNS(null, "cy", "50%");
        workerNode.setAttributeNS(null, "class", "node worker");
        this.dashboard.appendChild(workerNode);
        this.workers.splice(Math.floor(Math.random() * this.workers.length) - 1, 0, id)
        this.transfers[id] = {}
        this.update_worker_positions()
    }

    remove_worker(id) {
        let index = this.workers.indexOf(id);
        if (index > -1) {
            this.workers.splice(index, 1);
        }

        delete this.transfers[id]

        let worker = document.getElementById(id)
        this.dashboard.removeChild(worker)

        this.update_worker_positions()
    }

    update_worker_positions() {
        for (var i = 0; i < this.workers.length; i++) {
            let θ = (2 * Math.PI * i / this.workers.length)
            let r = 40
            let h = 50
            let k = 50
            let x = h + r * Math.cos(θ)
            let y = k + r * Math.sin(θ)
            gsap.to('#' + this.workers[i], { r: 15, cx: x + "%", cy: y + "%", duration: 0.25 });
        }
    }

    create_task(task, worker) {
        this.tasks[task] = { 'worker': worker }
    }

    start_task(worker_id, task_name, color) {
        this.create_task(task_name, worker_id)
        let worker = document.getElementById(worker_id)
        let scheduler = document.getElementById("scheduler")
        let taskTl = gsap.timeline({});
        taskTl.add(() => {
            this.fire_projectile(
                scheduler,
                worker,
                color)
        })
        taskTl.to('#' + worker_id, { fill: color, duration: 0.25 }, 0.5);
    }

    end_task(worker_id) {
        gsap.to('#' + worker_id, { fill: null, duration: 0.25 });
    }

    fire_projectile(start_element, end_element, color) {
        let projectileTl = gsap.timeline({});
        let arc = this.draw_arc(start_element, end_element, color, "projectile")
        projectileTl.add(() => { this.dashboard.insertBefore(arc, this.schedulerNode); }, 0)
        projectileTl.add(() => { this.dashboard.removeChild(arc) }, 1)
        projectileTl.play()
    }

    start_transfer(start_worker, end_worker) {
        let color = "rgba(255, 0, 0, .6)"
        let arc = this.draw_arc(document.getElementById(start_worker), document.getElementById(end_worker), color, "transfer")
        this.dashboard.insertBefore(arc, this.schedulerNode);
        this.transfers.push(arc)
        gsap.to('#' + start_worker, { fill: color, duration: 0.25 });
        gsap.to('#' + end_worker, { fill: color, duration: 0.25 });
        return arc;
    }

    end_transfer(start_worker, end_worker, arc) {
        this.dashboard.removeChild(arc)
        var index = this.transfers.indexOf(arc);
        if (index !== -1) this.transfers.splice(index, 1);
        gsap.to('#' + start_worker, { fill: null, duration: 0.25 });
        gsap.to('#' + end_worker, { fill: null, duration: 0.25 });
    }

    start_swap(worker) {
        let color = "#D67548"
        gsap.to('#' + worker, { fill: color, duration: 0.25 });
    }

    end_swap(worker) {
        gsap.to('#' + worker, { fill: null, duration: 0.25 });
    }

    kill_worker(worker) {
        let color = "rgba(0, 0, 0, 1)"
        gsap.to('#' + worker, { fill: color, duration: 0.25 });
    }

    reset() {
        for (var i = 0; i < this.workers.length; i++) {
            gsap.to('#' + this.workers[i], { fill: null, duration: 0.25 });
        }
        for (var arc in this.transfers) {
            this.dashboard.removeChild(tarc)
            var index = this.transfers.indexOf(arc);
            if (index !== -1) this.transfers.splice(index, 1);
        }
    }

    draw_arc(start_element, end_element, color, class_name) {
        let start_x = getAbsoluteXY(start_element)[0];
        let start_y = getAbsoluteXY(start_element)[1];
        let end_x = getAbsoluteXY(end_element)[0];
        let end_y = getAbsoluteXY(end_element)[1];
        let arc = document.createElementNS('http://www.w3.org/2000/svg', "path");
        arc.setAttributeNS(null, "id", class_name);
        arc.setAttributeNS(null, "class", class_name);
        arc.setAttributeNS(null, "stroke", color);

        // mid-point of line:
        var mpx = (start_x + end_x) * 0.5;
        var mpy = (start_y + end_y) * 0.5;

        // angle of perpendicular to line:
        var theta = Math.atan2(start_y - end_y, start_x - end_x) - Math.PI / 2;

        // distance of control point from mid-point of line:
        var offset = Math.random() * 50;
        if (Math.random() >= 0.5) {
            offset = -offset
        }

        // location of control point:
        var c1x = mpx + offset * Math.cos(theta);
        var c1y = mpy + offset * Math.sin(theta);

        // construct the command to draw a quadratic curve
        var curve = "M" + end_x + " " + end_y + " Q " + c1x + " " + c1y + " " + start_x + " " + start_y;
        arc.setAttribute("d", curve);
        return arc
    }
}

function getAbsoluteXY(element) {
    var box = element.getBoundingClientRect();
    var x = box.left + (box.width / 4);
    var y = box.top + (box.height / 4);
    return [x, y]
}

var dashboard

function websocket_url(s) {
    var l = window.location;
    return ((l.protocol === "https:") ? "wss://" : "ws://") + l.hostname + (((l.port != 80) && (l.port != 443)) ? ":" + l.port : "") + s;
}

function main() {
    console.log("Starting...")
    dashboard = new Dashboard()
    dashboard.add_scheduler()

    var ws = new WebSocket(websocket_url("/events"));
    ws.onopen = function () {
        ws.send(JSON.stringify({
            "name": "ping"
        }));
    };
    ws.onmessage = function (event) {
        dashboard.handle_event(JSON.parse(event.data))
    };
}

window.addEventListener('load', main)