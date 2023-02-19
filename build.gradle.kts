plugins {
    java
    application
}

group = "quannm"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.openpnp:opencv:4.5.5-1")

    implementation("commons-io:commons-io:2.11.0")
    implementation("com.google.guava:guava:31.1-jre")


//    implementation("org.jetbrains.kotlin:kotlinx-cli-jvm:0.3.5")
}

application {
    mainClass.set("quannm.Main")
}

tasks.test {
    useJUnitPlatform()
}

sourceSets {
    main {
        java {
            srcDir("src")
            exclude("**/*.txt")
        }
        resources {
            srcDir("src")
            exclude("*.java")
        }
    }
}
