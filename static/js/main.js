/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.canvas.width  = 513; // 8 * 64 + 1
        this.canvas.height = 513; // 8 * 64 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 513, 513);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 513, 513);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < 63; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 8,   0);
            this.ctx.lineTo((i + 1) * 8, 513);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(  0, (i + 1) * 8);
            this.ctx.lineTo(513, (i + 1) * 8);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput();
        $('#output td').text('').removeClass('success');
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 8;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    upload() {
	let file = document.querySelector('input[type=file]').files[0];  // 获取选择的文件，这里是图片类型
	let reader = new FileReader();
        reader.readAsDataURL(file); //读取文件并将文件以URL的形式保存在resulr属性中 base64格式
        reader.onload = function(e) { // 文件读取完成时触发
            let result = e.target.result; // base64格式图片地址
            var image = new Image();
	    image.src = result; // 设置image的地址为base64的地址
            image.onload = function(){
                var canvas = document.querySelector("#main");
                var context = canvas.getContext("2d");
                canvas.width = 513; // 设置canvas的画布宽度为图片宽度
                canvas.height = 513;
                // 等比例缩放图片
					var scale = 1;
					var canvas_req = 513; // 可以根据具体的要求去设定
					if (this.width > canvas_req || this.height > canvas_req) {
						if (this.width > this.height) {
							scale = canvas_req / this.width;
						}else {
							scale = canvas_req / this.height;
						}
					}
					context.width = this.width * scale;
					context.height = this.height * scale; // 计算等比缩小后图片
                context.drawImage(image, 0, 0, context.width, context.height); // 在canvas上绘制图片
                let dataUrl = canvas.toDataURL('image/jpeg', 0.92); // 0.92为压缩比，可根据需要设置，设置过小会影响图片质量
                                                                   // dataUrl 为压缩后的图片资源，可将其上传到服务器
            };
       };
 }
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 64, 64);
            console.log(img.width);

            console.log(img.height);
            var data = small.getImageData(0, 0, 64, 64).data;
            for (var i = 0; i < 64; i++) {
                for (var j = 0; j < 64; j++) {
                    var n = 4 * (i * 64 + j);
                    inputs[i * 64 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 3, i * 3, 3, 3);
                }
            }
            if (Math.min(...inputs) === 255) {
                return;
            }
            $.ajax({
                url: '/api/zhongwen',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(inputs),
                success: (data) => {
                    //测试数据
                //data.results = ['留','留','留','留','留','留','留','留','留','留']
                    for (let j = 0; j < 5; j++) {
                        var value = data.results[j];
                        let find = $('#output tr').eq(j + 1).find('td');
                        find.eq(0).text(value);
                        value = data.results[j+5];
                        find.eq(1).text(value);
                    }


                }
            });
        };
        img.src = this.canvas.toDataURL();
    }
}

$(() => {
    var main = new Main();
    $('#clear').click(() => {
        main.initialize();
    });
    $('#con').click(() => {
        main.upload();
        setTimeout(function(){main.drawInput();},"1000");
    });
});
