package SuperMDS;

import java.io.InputStream;
import java.net.URL;
/**
 *
 * @author Sean Phillips
 */
public class StyleResourceProvider {

    public static InputStream getResourceAsStream(String name) {
        return StyleResourceProvider.class.getResourceAsStream(name);
    }

    public static URL getResource(String name) {
        return StyleResourceProvider.class.getResource(name);
    }

}