package SuperMDS;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.scene.Scene;
import javafx.scene.input.DataFormat;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * @author Sean Phillips
 */
public enum ResourceUtils {
    INSTANCE;
    private static final Logger LOG = LoggerFactory.getLogger(ResourceUtils.class);


    public static boolean canDragOver(DragEvent event) {
        Dragboard db = event.getDragboard();
        DataFormat dataFormat = DataFormat.lookupMimeType("application/x-java-file-list");
        try {
            if (db.hasFiles() || db.hasContent(dataFormat)) {
                List<File> files = db.getFiles();
                //workaround for Swing JFXPanel
                if (db.hasContent(dataFormat)) {
                    //Swing containers require a registered mime type
                    //since we don't have that, we need to accept the drag
                    event.acceptTransferModes(TransferMode.COPY);
                    return true;
                }
            } else {
                event.consume();
            }
        } catch (Exception ex) {
            LOG.error(null, ex);
            event.consume();
        }
        return false;
    }

    /**
     * Any time a drop event occurs this attempts to process the object.
     *
     * @param event DragEvent.DragDropped
     * @param scene
     */
    public static void onDragDropped(DragEvent event, Scene scene) {
        Dragboard db = event.getDragboard();
        if (db.hasFiles()) {
            final List<File> files = db.getFiles();
            Task task = new Task() {
                @Override
                protected Void call() throws Exception {
                    for (File file : files) {
//                        try {
////                            if (CocoAnnotationFile.isCocoAnnotationFile(file)) {
////                                CocoAnnotationFile cocoFile = new CocoAnnotationFile(file.getAbsolutePath(), true);
////                                Platform.runLater(() -> scene.getRoot().fireEvent(
////                                    new ImageEvent(ImageEvent.NEW_COCO_ANNOTATION, cocoFile.cocoObject)));
////                            }
//                        } catch (IOException ex) {
//                            LOG.error(null, ex);
//                        }
                    }
                    return null;
                }
            };
            Thread t = new Thread(task);
            t.setDaemon(true);
            t.start();
            event.setDropCompleted(true);
            event.consume();
        }
    }


}
