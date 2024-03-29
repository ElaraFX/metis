#include "mainwindow.h"
#include "qtpropertymanager.h"
#include "qteditorfactory.h"
#include "qttreepropertybrowser.h"
#include <QMatrix>
#include <QMouseEvent>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDockWidget>
#include <QToolBar>
#include <qspinbox.h>
#include <qlabel.h>

#include "CUDAWindow.h"

void CanvasView::contentsMousePressEvent(QMouseEvent* event)
{
    handleMouseClickEvent(event);
}

void CanvasView::contentsMouseDoubleClickEvent(QMouseEvent* event)
{
    handleMouseClickEvent(event);
}

void CanvasView::handleMouseClickEvent(QMouseEvent* event)
{
    QPoint p = inverseWorldMatrix().map(event->pos());
    QtCanvasItemList l = canvas()->collisions(p);
    moving = 0;
    if (!l.isEmpty())
        moving = l.first();
    moving_start = p;
    emit itemClicked(moving);
}

void CanvasView::contentsMouseMoveEvent(QMouseEvent* event)
{
    if (moving) {
	QPoint p = inverseWorldMatrix().map(event->pos());
	moving->moveBy(p.x() - moving_start.x(), p.y() - moving_start.y());
	moving_start = p;
	canvas()->update();
        emit itemMoved(moving);
    }
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
	this->setWindowTitle(QString::fromLocal8Bit("闪渲科技 - Metis渲染器 V0.1"));
    fileToolBar = addToolBar(tr("File"));
    statusBar();
    QMenu *editMenu = menuBar()->addMenu(tr("File"));
    QMenu *newObjectMenu = editMenu->addMenu(tr("New Object"));
    menuBar()->addMenu(tr("Edit"));
    menuBar()->addMenu(tr("Project"));
    menuBar()->addMenu(tr("Create"));
    menuBar()->addMenu(tr("Align"));
    menuBar()->addMenu(tr("Window"));
    menuBar()->addMenu(tr("Help"));

    QAction *newRectangleAction = new QAction(tr("Rectangle"), this);
    connect(newRectangleAction, SIGNAL(triggered(bool)), this, SLOT(newRectangle()));
    newObjectMenu->addAction(newRectangleAction);
    //fileToolBar->addAction(newRectangleAction);

    QAction *newLineAction = new QAction(tr("Line"), this);
    connect(newLineAction, SIGNAL(triggered(bool)), this, SLOT(newLine()));
    newObjectMenu->addAction(newLineAction);
    //fileToolBar->addAction(newLineAction);

    QAction *newEllipseAction = new QAction(tr("Ellipse"), this);
    connect(newEllipseAction, SIGNAL(triggered(bool)), this, SLOT(newEllipse()));
    newObjectMenu->addAction(newEllipseAction);
    //fileToolBar->addAction(newEllipseAction);

    QAction *newTextAction = new QAction(tr("Text"), this);
    connect(newTextAction, SIGNAL(triggered(bool)), this, SLOT(newText()));
    newObjectMenu->addAction(newTextAction);
    //fileToolBar->addAction(newTextAction);

    deleteAction = new QAction(tr("Delete Object"), this);
    connect(deleteAction, SIGNAL(triggered(bool)), this, SLOT(deleteObject()));
    editMenu->addAction(deleteAction);
    //fileToolBar->addAction(deleteAction);

    QAction *clearAction = new QAction(tr("Clear All"), this);
    connect(clearAction, SIGNAL(triggered(bool)), this, SLOT(clearAll()));
    editMenu->addAction(clearAction);
    //fileToolBar->addAction(clearAction);

    QAction *fillAction = new QAction(tr("Fill View"), this);
    connect(fillAction, SIGNAL(triggered(bool)), this, SLOT(fillView()));
    editMenu->addAction(fillAction);
    //fileToolBar->addAction(fillAction);

    doubleManager = new QtDoublePropertyManager(this);
    stringManager = new QtStringPropertyManager(this);
    colorManager = new QtColorPropertyManager(this);
    fontManager = new QtFontPropertyManager(this);
    pointManager = new QtPointPropertyManager(this);
    sizeManager = new QtSizePropertyManager(this);

    connect(doubleManager, SIGNAL(valueChanged(QtProperty *, double)),
                this, SLOT(valueChanged(QtProperty *, double)));
    connect(stringManager, SIGNAL(valueChanged(QtProperty *, const QString &)),
                this, SLOT(valueChanged(QtProperty *, const QString &)));
    connect(colorManager, SIGNAL(valueChanged(QtProperty *, const QColor &)),
                this, SLOT(valueChanged(QtProperty *, const QColor &)));
    connect(fontManager, SIGNAL(valueChanged(QtProperty *, const QFont &)),
                this, SLOT(valueChanged(QtProperty *, const QFont &)));
    connect(pointManager, SIGNAL(valueChanged(QtProperty *, const QPoint &)),
                this, SLOT(valueChanged(QtProperty *, const QPoint &)));
    connect(sizeManager, SIGNAL(valueChanged(QtProperty *, const QSize &)),
                this, SLOT(valueChanged(QtProperty *, const QSize &)));

    QtDoubleSpinBoxFactory *doubleSpinBoxFactory = new QtDoubleSpinBoxFactory(this);
    QtCheckBoxFactory *checkBoxFactory = new QtCheckBoxFactory(this);
    QtSpinBoxFactory *spinBoxFactory = new QtSpinBoxFactory(this);
    QtLineEditFactory *lineEditFactory = new QtLineEditFactory(this);
    QtEnumEditorFactory *comboBoxFactory = new QtEnumEditorFactory(this);

    canvas = new QtCanvas(1280, 768);
    cudaWindow = new TriangleWindow(this);
    //canvasView = new CanvasView(canvas, this);
    setCentralWidget(cudaWindow);

    QDockWidget *dock = new QDockWidget(this);
    addDockWidget(Qt::RightDockWidgetArea, dock);

	m_filterSizeSpin = new QSpinBox();
	m_variancePosSpin = new QDoubleSpinBox();
	m_varianceColSpin = new QDoubleSpinBox();
	m_colorsatSpin = new QDoubleSpinBox();
	m_exposureValSpin = new QDoubleSpinBox();
	m_whitePointSpin = new QDoubleSpinBox();
	m_shadowsSpin = new QDoubleSpinBox();
	m_midtonesSpin = new QDoubleSpinBox();
	m_highlightsSpin = new QDoubleSpinBox();
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("去噪窗口")));
	fileToolBar->addWidget(m_filterSizeSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("位置影响因子")));
	fileToolBar->addWidget(m_variancePosSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("颜色影响因子")));
	fileToolBar->addWidget(m_varianceColSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("颜色饱和度")));
	fileToolBar->addWidget(m_colorsatSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("曝光指数")));
	fileToolBar->addWidget(m_exposureValSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("白平衡")));
	fileToolBar->addWidget(m_whitePointSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("暗区亮度")));
	fileToolBar->addWidget(m_shadowsSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("普通亮度")));
	fileToolBar->addWidget(m_midtonesSpin);
	fileToolBar->addWidget(new QLabel(QString::fromLocal8Bit("高光区亮度")));
	fileToolBar->addWidget(m_highlightsSpin);

	m_filterSizeSpin->setValue(15);
	m_variancePosSpin->setRange(0, 99999);
	m_variancePosSpin->setValue(100);
	m_varianceColSpin->setRange(0, 99999);
	m_varianceColSpin->setValue(200);
	m_colorsatSpin->setRange(0, 99999);
	m_colorsatSpin->setValue(1);
	m_exposureValSpin->setRange(-100, 100);
	m_exposureValSpin->setValue(0);
	m_whitePointSpin->setRange(0, 40000);
	m_whitePointSpin->setValue(6500);
	m_shadowsSpin->setRange(0, 1);
	m_shadowsSpin->setValue(0.2);
	m_midtonesSpin->setRange(0.01, 4);
	m_midtonesSpin->setValue(1);
	m_highlightsSpin->setRange(0, 1);
	m_highlightsSpin->setValue(0.2);
	connect(m_filterSizeSpin, SIGNAL(valueChanged(int)), cudaWindow, SLOT(slotWindowSizeChanged(int)));
	connect(m_variancePosSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotVariancPosChanged(double)));
	connect(m_varianceColSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotVariancColChanged(double)));
	connect(m_colorsatSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotColorsatChanged(double)));
	connect(m_exposureValSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotExposureValChanged(double)));
	connect(m_whitePointSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotWhitepointChanged(double)));
	connect(m_shadowsSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotShadowsChanged(double)));
	connect(m_midtonesSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotMidtonesChanged(double)));
	connect(m_highlightsSpin, SIGNAL(valueChanged(double)), cudaWindow, SLOT(slotHighlightsChanged(double)));

    propertyEditor = new QtTreePropertyBrowser(dock);
    propertyEditor->setFactoryForManager(doubleManager, doubleSpinBoxFactory);
    propertyEditor->setFactoryForManager(stringManager, lineEditFactory);
    propertyEditor->setFactoryForManager(colorManager->subIntPropertyManager(), spinBoxFactory);
    propertyEditor->setFactoryForManager(fontManager->subIntPropertyManager(), spinBoxFactory);
    propertyEditor->setFactoryForManager(fontManager->subBoolPropertyManager(), checkBoxFactory);
    propertyEditor->setFactoryForManager(fontManager->subEnumPropertyManager(), comboBoxFactory);
    propertyEditor->setFactoryForManager(pointManager->subIntPropertyManager(), spinBoxFactory);
    propertyEditor->setFactoryForManager(sizeManager->subIntPropertyManager(), spinBoxFactory);
    dock->setWidget(propertyEditor);

    currentItem = 0;

    //connect(canvasView, SIGNAL(itemClicked(QtCanvasItem *)),
    //        this, SLOT(itemClicked(QtCanvasItem *)));
    //connect(canvasView, SIGNAL(itemMoved(QtCanvasItem *)),
    //        this, SLOT(itemMoved(QtCanvasItem *)));

    //fillView();
    itemClicked(0);

	cudaWindow->setFocus();
}

void MainWindow::newRectangle()
{
    QtCanvasItem *item = addRectangle();
    canvas->update();
    itemClicked(item);
}

void MainWindow::newEllipse()
{
    QtCanvasItem *item = addEllipse();
    canvas->update();
    itemClicked(item);
}

void MainWindow::newLine()
{
    QtCanvasItem *item = addLine();
    canvas->update();
    itemClicked(item);
}

void MainWindow::newText()
{
    QtCanvasItem *item = addText();
    canvas->update();
    itemClicked(item);
}

void MainWindow::deleteObject()
{
    if (!currentItem)
        return;

    delete currentItem;
    itemClicked(0);
    canvas->update();
}

void MainWindow::clearAll()
{
    QtCanvasItemList list = canvas->allItems();
    qDeleteAll(list);
    itemClicked(0);
    canvas->update();
}

void MainWindow::fillView()
{
    for (int i = 0; i < 10; i++) {
        addRectangle();
        addEllipse();
        addLine();
        addText();
    }
    canvas->update();
}

QtCanvasItem *MainWindow::addRectangle()
{
    QtCanvasPolygonalItem *item = new QtCanvasRectangle(rand() % canvas->width(),
                rand() % canvas->height(), 50, 50, canvas);
    int z = rand() % 256;
    item->setBrush(QColor(rand() % 32 * 8, rand() % 32 * 8, rand() % 32 * 8));
    item->setPen(QPen(QColor(rand() % 32*8, rand() % 32*8, rand() % 32*8), 4));
    item->setZ(z);
    item->show();
    return item;
}

QtCanvasItem *MainWindow::addEllipse()
{
    QtCanvasPolygonalItem *item = new QtCanvasEllipse(50, 50, canvas);
    item->setBrush(QColor(rand() % 32 * 8, rand() % 32 * 8, rand() % 32 * 8));
    item->move(rand() % canvas->width(), rand() % canvas->height());
    item->setZ(rand() % 256);
    item->show();
    return item;
}

QtCanvasItem *MainWindow::addLine()
{
    QtCanvasLine *item = new QtCanvasLine(canvas);
    item->setPoints(0, 0, rand() % canvas->width() - canvas->width() / 2,
                rand() % canvas->height() - canvas->height() / 2);
    item->move(rand() % canvas->width(), rand() % canvas->height());
    item->setPen(QPen(QColor(rand() % 32*8, rand() % 32*8, rand() % 32*8), 6));
    item->setZ(rand() % 256);
    item->show();
    return item;
}

QtCanvasItem *MainWindow::addText()
{
    QtCanvasText *item = new QtCanvasText(canvas);
    item->setText(tr("Text"));
    item->setColor(QColor(rand() % 32*8, rand() % 32*8, rand() % 32*8));
    item->move(rand() % canvas->width(), rand() % canvas->height());
    item->setZ(rand() % 256);
    item->show();
    return item;
}

void MainWindow::itemMoved(QtCanvasItem *item)
{
    if (item != currentItem)
        return;

    doubleManager->setValue(idToProperty[QLatin1String("xpos")], item->x());
    doubleManager->setValue(idToProperty[QLatin1String("ypos")], item->y());
    doubleManager->setValue(idToProperty[QLatin1String("zpos")], item->z());
}

void MainWindow::updateExpandState()
{
    QList<QtBrowserItem *> list = propertyEditor->topLevelItems();
    QListIterator<QtBrowserItem *> it(list);
    while (it.hasNext()) {
        QtBrowserItem *item = it.next();
        QtProperty *prop = item->property();
        idToExpanded[propertyToId[prop]] = propertyEditor->isExpanded(item);
    }
}

void MainWindow::itemClicked(QtCanvasItem *item)
{
    updateExpandState();

    QMap<QtProperty *, QString>::ConstIterator itProp = propertyToId.constBegin();
    while (itProp != propertyToId.constEnd()) {
        delete itProp.key();
        itProp++;
    }
    propertyToId.clear();
    idToProperty.clear();

    currentItem = item;
    if (!currentItem) {
        deleteAction->setEnabled(false);
        return;
    }

    deleteAction->setEnabled(true);

    QtProperty *property;

    property = doubleManager->addProperty(tr("Position X"));
    doubleManager->setRange(property, 0, canvas->width());
    doubleManager->setValue(property, item->x());
    addProperty(property, QLatin1String("xpos"));

    property = doubleManager->addProperty(tr("Position Y"));
    doubleManager->setRange(property, 0, canvas->height());
    doubleManager->setValue(property, item->y());
    addProperty(property, QLatin1String("ypos"));

    property = doubleManager->addProperty(tr("Position Z"));
    doubleManager->setRange(property, 0, 256);
    doubleManager->setValue(property, item->z());
    addProperty(property, QLatin1String("zpos"));

    if (item->rtti() == QtCanvasItem::Rtti_Rectangle) {
        QtCanvasRectangle *i = (QtCanvasRectangle *)item;

        property = colorManager->addProperty(tr("Brush Color"));
        colorManager->setValue(property, i->brush().color());
        addProperty(property, QLatin1String("brush"));

        property = colorManager->addProperty(tr("Pen Color"));
        colorManager->setValue(property, i->pen().color());
        addProperty(property, QLatin1String("pen"));

        property = sizeManager->addProperty(tr("Size"));
        sizeManager->setValue(property, i->size());
        addProperty(property, QLatin1String("size"));
    } else if (item->rtti() == QtCanvasItem::Rtti_Line) {
        QtCanvasLine *i = (QtCanvasLine *)item;

        property = colorManager->addProperty(tr("Pen Color"));
        colorManager->setValue(property, i->pen().color());
        addProperty(property, QLatin1String("pen"));

        property = pointManager->addProperty(tr("Vector"));
        pointManager->setValue(property, i->endPoint());
        addProperty(property, QLatin1String("endpoint"));
    } else if (item->rtti() == QtCanvasItem::Rtti_Ellipse) {
        QtCanvasEllipse *i = (QtCanvasEllipse *)item;

        property = colorManager->addProperty(tr("Brush Color"));
        colorManager->setValue(property, i->brush().color());
        addProperty(property, QLatin1String("brush"));

        property = sizeManager->addProperty(tr("Size"));
        sizeManager->setValue(property, QSize(i->width(), i->height()));
        sizeManager->setRange(property, QSize(0, 0), QSize(1000, 1000));
        addProperty(property, QLatin1String("size"));
    } else if (item->rtti() == QtCanvasItem::Rtti_Text) {
        QtCanvasText *i = (QtCanvasText *)item;

        property = colorManager->addProperty(tr("Color"));
        colorManager->setValue(property, i->color());
        addProperty(property, QLatin1String("color"));

        property = stringManager->addProperty(tr("Text"));
        stringManager->setValue(property, i->text());
        addProperty(property, QLatin1String("text"));

        property = fontManager->addProperty(tr("Font"));
        fontManager->setValue(property, i->font());
        addProperty(property, QLatin1String("font"));
    }
}

void MainWindow::addProperty(QtProperty *property, const QString &id)
{
    propertyToId[property] = id;
    idToProperty[id] = property;
    QtBrowserItem *item = propertyEditor->addProperty(property);
    if (idToExpanded.contains(id))
        propertyEditor->setExpanded(item, idToExpanded[id]);
}

void MainWindow::valueChanged(QtProperty *property, double value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (id == QLatin1String("xpos")) {
        currentItem->setX(value);
    } else if (id == QLatin1String("ypos")) {
        currentItem->setY(value);
    } else if (id == QLatin1String("zpos")) {
        currentItem->setZ(value);
    }
    canvas->update();
}

void MainWindow::valueChanged(QtProperty *property, const QString &value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (id == QLatin1String("text")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Text) {
            QtCanvasText *i = (QtCanvasText *)currentItem;
            i->setText(value);
        }
    }
    canvas->update();
}

void MainWindow::valueChanged(QtProperty *property, const QColor &value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (id == QLatin1String("color")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Text) {
            QtCanvasText *i = (QtCanvasText *)currentItem;
            i->setColor(value);
        }
    } else if (id == QLatin1String("brush")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Rectangle ||
                currentItem->rtti() == QtCanvasItem::Rtti_Ellipse) {
            QtCanvasPolygonalItem *i = (QtCanvasPolygonalItem *)currentItem;
            QBrush b = i->brush();
            b.setColor(value);
            i->setBrush(b);
        }
    } else if (id == QLatin1String("pen")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Rectangle ||
                currentItem->rtti() == QtCanvasItem::Rtti_Line) {
            QtCanvasPolygonalItem *i = (QtCanvasPolygonalItem *)currentItem;
            QPen p = i->pen();
            p.setColor(value);
            i->setPen(p);
        }
    }
    canvas->update();
}

void MainWindow::valueChanged(QtProperty *property, const QFont &value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (id == QLatin1String("font")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Text) {
            QtCanvasText *i = (QtCanvasText *)currentItem;
            i->setFont(value);
        }
    }
    canvas->update();
}

void MainWindow::valueChanged(QtProperty *property, const QPoint &value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (currentItem->rtti() == QtCanvasItem::Rtti_Line) {
        QtCanvasLine *i = (QtCanvasLine *)currentItem;
        if (id == QLatin1String("endpoint")) {
            i->setPoints(i->startPoint().x(), i->startPoint().y(), value.x(), value.y());
        }
    }
    canvas->update();
}

void MainWindow::valueChanged(QtProperty *property, const QSize &value)
{
    if (!propertyToId.contains(property))
        return;

    if (!currentItem)
        return;

    QString id = propertyToId[property];
    if (id == QLatin1String("size")) {
        if (currentItem->rtti() == QtCanvasItem::Rtti_Rectangle) {
            QtCanvasRectangle *i = (QtCanvasRectangle *)currentItem;
            i->setSize(value.width(), value.height());
        } else if (currentItem->rtti() == QtCanvasItem::Rtti_Ellipse) {
            QtCanvasEllipse *i = (QtCanvasEllipse *)currentItem;
            i->setSize(value.width(), value.height());
        }
    }
    canvas->update();
}

