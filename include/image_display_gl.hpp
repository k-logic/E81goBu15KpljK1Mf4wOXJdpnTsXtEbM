#pragma once
#include <GLFW/glfw3.h>
#include <GLES2/gl2.h>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace image_display {

static std::thread display_thread;
static std::atomic<bool> running{false};
static int gWidth = 1280, gHeight = 720;  // デフォルトサイズ

// フレーム共有用
static std::vector<unsigned char> frameBuffer;
static std::mutex frameMutex;
static std::condition_variable frameCV;
static bool newFrameAvailable = false;

// OpenGL用
static GLuint shaderProgram = 0;
static GLuint textureID = 0;
static GLuint vbo = 0;

// 頂点+UV
static const GLfloat vertices[] = {
    // pos   // tex
    -1.f, -1.f,  0.f, 0.f,
     1.f, -1.f,  1.f, 0.f,
    -1.f,  1.f,  0.f, 1.f,
     1.f,  1.f,  1.f, 1.f,
};

// シェーダ作成
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    return shader;
}

void display_loop() {
    if (!glfwInit()) return;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(gWidth, gHeight, "Decoder Display (GLES2)", nullptr, nullptr);
    if (!window) { glfwTerminate(); return; }
    glfwMakeContextCurrent(window);

    const char* vsrc =
        "attribute vec2 aPos;"
        "attribute vec2 aTexCoord;"
        "varying vec2 vTexCoord;"
        "void main(){"
        " gl_Position=vec4(aPos,0.0,1.0);"
        " vTexCoord=aTexCoord;"
        "}";

    const char* fsrc =
        "precision mediump float;"
        "varying vec2 vTexCoord;"
        "uniform sampler2D uTexture;"
        "void main(){"
        " gl_FragColor=texture2D(uTexture,vTexCoord);"
        "}";

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // VBO作成
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLint aPos = glGetAttribLocation(shaderProgram, "aPos");
    GLint aTex = glGetAttribLocation(shaderProgram, "aTexCoord");
    glEnableVertexAttribArray(aPos);
    glEnableVertexAttribArray(aTex);
    glVertexAttribPointer(aPos, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
    glVertexAttribPointer(aTex, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));

    // テクスチャ作成
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    while (running && !glfwWindowShouldClose(window)) {
        std::unique_lock<std::mutex> lk(frameMutex);
        frameCV.wait(lk, [] { return newFrameAvailable || !running; });
        if (!running) break;

        std::vector<unsigned char> localFrame = frameBuffer;
        newFrameAvailable = false;
        lk.unlock();

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     gWidth, gHeight, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, localFrame.data());

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

// ディスプレイサイズ設定（必要なら呼ぶ）
void set_display_size(int w, int h) {
    gWidth = w;
    gHeight = h;
}

// スレッド開始
void start_display_thread() {
    running = true;
    display_thread = std::thread(display_loop);
}

// 入力: CHW形式 (3 × H × W), 出力: HWC形式 (h × w × 3)
void update_frame(const uint8_t* chw, int c, int h, int w) {
    std::lock_guard<std::mutex> lk(frameMutex);
    frameBuffer.resize(h * w * c);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx_rgb = (y * w + x) * c;

            // chw: [0]=B, [1]=G, [2]=R と仮定
            frameBuffer[idx_rgb + 0] = chw[2 * h * w + y * w + x]; // R
            frameBuffer[idx_rgb + 1] = chw[1 * h * w + y * w + x]; // G
            frameBuffer[idx_rgb + 2] = chw[0 * h * w + y * w + x]; // B
        }
    }

    newFrameAvailable = true;
    frameCV.notify_one();
}

// 停止
void stop_display_thread() {
    running = false;
    frameCV.notify_all();
    if (display_thread.joinable()) display_thread.join();
}

}
