#include <QApplication>
#include "mainwindow.h"
#include "CUDAWindow.h"
#include <QStyleFactory>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    QApplication::setStyle(QStyleFactory::create("Fusion"));
    QPalette p;
    p = qApp->palette();
    p.setColor(QPalette::Window, QColor(53,53,53));
    p.setColor(QPalette::Button, QColor(53,53,53));
    p.setColor(QPalette::Highlight, QColor(142,45,197));
    p.setColor(QPalette::ButtonText, QColor(255,255,255));
    qApp->setPalette(p);

    MainWindow mw;
    mw.resize(1500, 800);
    mw.show();

    //QSurfaceFormat format;
    //format.setSamples(16);

    //TriangleWindow window;
    //window.setFormat(format);
    //window.resize(800, 600);
    //window.show();

    //window.setAnimating(true);

    return app.exec();
}

